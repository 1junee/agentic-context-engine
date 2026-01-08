#!/usr/bin/env python3
"""
Simplest possible ACE example with ACELiteLLM.

This shows the minimal code needed to use ACE with a production LLM.
"""

import dataclasses
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment

REFLECTION_WINDOW = 3
SKILLBOOK_PATH = Path(__file__).with_name("simple_ace_skillbook.json")

# Load environment variables
load_dotenv()

def format_payload(payload):
    """Pretty-print dict-like payloads as JSON."""
    if dataclasses.is_dataclass(payload):
        payload = dataclasses.asdict(payload)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def print_header(title, char="="):
    """Print a visual header for a major section."""
    line = char * len(title)
    print(f"\n{title}\n{line}")


def print_section(title, char="-"):
    """Print a section header."""
    line = char * len(title)
    print(f"\n{title}\n{line}")


def print_skillbook(skillbook, label):
    """Print the current skillbook state."""
    print_header(label)
    print(f"skills: {len(skillbook.skills())}")
    print(skillbook)


def print_step_outputs(
    agent_output, env_result, reflection, skill_manager_output, index
):
    """Print Agent/Environment/Reflector/SkillManager outputs for a single sample."""
    print_header(f"SAMPLE {index}")

    print_section("Agent")
    print(format_payload(agent_output.model_dump()))

    print_section("Environment")
    print(format_payload(env_result))

    print_section("Reflector")
    print(format_payload(reflection.model_dump()))
    
    print_section("SkillManager")
    print(format_payload(skill_manager_output.update.to_json()))


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
    if SKILLBOOK_PATH.exists():
        print_header("Skillbook Load")
        print(f"loading from: {SKILLBOOK_PATH}")
        agent = ACELiteLLM(
            model="deepseek/deepseek-chat",
            skillbook_path=str(SKILLBOOK_PATH),
        )
        print(f"loaded strategies: {len(agent.skillbook.skills())}")
    else:
        agent = ACELiteLLM(model="deepseek/deepseek-chat")

    # 2. Create training samples
    samples = [
        Sample(question="What is the result of 17 * 23?", ground_truth="391"),
        Sample(question="Solve for x in 3x - 7 = 20.", ground_truth=None),
        Sample(question="Convert binary 110011 to decimal.", ground_truth="51"),
        Sample(
            question="As a non-major, outline a beginner roadmap to learn AI basics.",
            ground_truth=None,
        ),
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
            agent_output, env_result, reflection, skill_manager_output, step_index
        )
        print_skillbook(agent.skillbook, f"Skillbook after sample {step_index}")

    # 4. Check results
    print_header("Training Summary")
    print(f"samples: {len(results)}")
    print(f"strategies: {len(agent.skillbook.skills())}")

    # Show a few learned strategies
    print_section("Learned Skills")
    for skill in agent.skillbook.skills()[:3]:
        print(f"- {skill.content}")

    # 4.5 Save learned skillbook
    agent.save_skillbook(str(SKILLBOOK_PATH))
    print_section("Skillbook Save")
    print(f"saved to: {SKILLBOOK_PATH}")

    # 5. Test with new question (show Agent output)
    test_question = "What is 5 + 3?"
    test_output = agent.agent.generate(
        question=test_question,
        context="",
        skillbook=agent.skillbook,
    )
    print_header("Test Run")
    print(f"question: {test_question}")
    print_section("Agent")
    print(format_payload(test_output.model_dump()))
    print(f"\nanswer: {test_output.final_answer}")


if __name__ == "__main__":
    main()
