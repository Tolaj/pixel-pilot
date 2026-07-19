"""Perceive-think-execute agent loop."""

import time
from perception import parse_screen, set_provider
from agent.llm import start_server, stop_server, get_next_action
from agent.executor import execute_action

MAX_STEPS = 20
SCREENSHOT_DELAY = 1.5


def run(task: str, provider: str = "moondream"):
    print(f"\nTask: {task}")
    print("=" * 50)

    set_provider(provider)
    start_server()

    history = []
    step = 0

    try:
        while step < MAX_STEPS:
            step += 1
            print(f"\n--- Step {step}/{MAX_STEPS} ---")

            print("Taking screenshot and parsing screen...")
            result = parse_screen()
            elements = result["elements"]
            screenshot = result["screenshot"]
            print(f"Found {len(elements)} elements in {result['parse_time']}s")

            print("Asking agent for next action...")
            action = get_next_action(
                task=task,
                screenshot=screenshot,
                elements=elements,
                history=history,
            )
            print(f"Thought: {action.get('thought', '')}")
            print(
                f"Action: {action.get('action')} | element_id={action.get('element_id')} | text={action.get('text')} | keys={action.get('keys')}"
            )

            if action.get("action") == "finished" or action.get("finished"):
                print("\nTask completed!")
                break

            try:
                result_msg = execute_action(action, elements)
                print(f"Executed: {result_msg}")
                history.append(f"{action.get('action')}: {result_msg}")
            except Exception as e:
                print(f"Execution error: {e}")
                history.append(f"error: {e}")

            time.sleep(SCREENSHOT_DELAY)

        else:
            print(f"\nReached max steps ({MAX_STEPS}), stopping.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        stop_server()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        task = input("What do you want me to do? ")
    else:
        task = " ".join(sys.argv[1:])
    run(task)
