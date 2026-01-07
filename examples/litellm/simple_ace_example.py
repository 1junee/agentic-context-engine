#!/usr/bin/env python3
"""
Simplest possible ACE example with ACELiteLLM.

This shows the minimal code needed to use ACE with a production LLM.
"""

import json
import os
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment

REFLECTION_WINDOW = 3

# Load environment variables
load_dotenv()

def print_skillbook(skillbook, label):
    """Print the current skillbook state."""
    print(f"\n[{label}]")
    print(skillbook)


def print_step_outputs(agent_output, reflection, skill_manager_output, index):
    """Print Agent/Reflector/SkillManager outputs for a single sample."""
    print(f"\n--- SAMPLE {index} ---")
    print("[Agent]")
    print(agent_output.model_dump())
    print("[Reflector]")
    print(reflection.model_dump())
    print("[SkillManager]")
    print(skill_manager_output.update.to_json())


def build_question_context(sample, env_result):
    parts = [
        f"question: {sample.question}",
        f"context: {sample.context}",
        f"metadata: {json.dumps(sample.metadata)}",
        f"feedback: {env_result.feedback}",
        f"ground_truth: {env_result.ground_truth}",
    ]
    return "\n".join(parts)


def progress_string(epoch, total_epochs, step, total_steps):
    return f"epoch {epoch}/{total_epochs} - sample {step}/{total_steps}"


def update_recent_reflections(recent_reflections, reflection):
    serialized = json.dumps(reflection.raw, ensure_ascii=False)
    recent_reflections.append(serialized)
    if len(recent_reflections) > REFLECTION_WINDOW:
        del recent_reflections[:-REFLECTION_WINDOW]


def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # 1. Create ACE agent (bundles all components with v2.1 prompts)
    # agent = ACELiteLLM(model="claude-sonnet-4-5-20250929")
    agent = ACELiteLLM(model="deepseek/deepseek-chat")

    # 2. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 3. Train the agent
    environment = SimpleEnvironment()
    results = []
    recent_reflections = []
    total_steps = len(samples)
    epoch = 1
    total_epochs = 1

    for step_index, sample in enumerate(samples, 1):
        print_skillbook(agent.skillbook, f"Skillbook before sample {step_index}")

        agent_output = agent.agent.generate(
            question=sample.question,
            context=sample.context,
            skillbook=agent.skillbook,
            reflection="\n---\n".join(recent_reflections),
            sample=sample,
        )
        env_result = environment.evaluate(sample, agent_output)
        reflection = agent.reflector.reflect(
            question=sample.question,
            agent_output=agent_output,
            skillbook=agent.skillbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
        )

        for tag in reflection.skill_tags:
            try:
                agent.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                pass

        update_recent_reflections(recent_reflections, reflection)

        skill_manager_output = agent.skill_manager.update_skills(
            reflection=reflection,
            skillbook=agent.skillbook,
            question_context=build_question_context(sample, env_result),
            progress=progress_string(epoch, total_epochs, step_index, total_steps),
        )

        agent.skillbook.apply_update(skill_manager_output.update)
        results.append((agent_output, reflection, skill_manager_output))

        print_step_outputs(
            agent_output, reflection, skill_manager_output, step_index
        )
        print_skillbook(agent.skillbook, f"Skillbook after sample {step_index}")

    # 4. Check results
    print(f"Trained on {len(results)} samples")
    print(f"Skillbook now has {len(agent.skillbook.skills())} strategies")

    # Show a few learned strategies
    for skill in agent.skillbook.skills()[:3]:
        print(f"\nLearned: {skill.content}")

    # 5. Test with new question (show Agent output)
    test_question = "What is 5 + 3?"
    test_output = agent.agent.generate(
        question=test_question,
        context="",
        skillbook=agent.skillbook,
    )
    print(f"\nTest question: {test_question}")
    print("[Agent]")
    print(test_output.model_dump())
    print(f"Answer: {test_output.final_answer}")


if __name__ == "__main__":
    main()
