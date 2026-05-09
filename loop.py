# loop.py
import time
from perception import parse_screen, set_provider
from agent import start_server, stop_server, get_next_action
from executor import execute_action

set_provider("moondream")  # "ax" or "moondream" / "omniparser"

MAX_STEPS = 20
SCREENSHOT_DELAY = 1.5  # seconds to wait after action before next screenshot


def run(task: str):
    print(f"\nTask: {task}")
    print("=" * 50)

    start_server()

    history = []
    step = 0

    try:
        while step < MAX_STEPS:
            step += 1
            print(f"\n--- Step {step}/{MAX_STEPS} ---")

            # perceive
            print("Taking screenshot and parsing screen...")
            result = parse_screen()
            elements = result["elements"]
            screenshot = result["screenshot"]
            print(f"Found {len(elements)} elements in {result['parse_time']}s")

            # think
            print("Asking agent for next action...")
            action = get_next_action(
                task=task,
                screenshot=screenshot,
                elements=elements,
                history=history,
            )
            print(f"Thought: {action.get('thought', '')}")
            print(
                f"Action: {action.get('action')} | element_id={action.get('element_id')} | text={action.get('text')} | keys={action.get('keys')} | key={action.get('key')}"
            )

            # check if done
            if action.get("action") == "finished" or action.get("finished"):
                print("\nTask completed!")
                break

            # execute
            try:
                result_msg = execute_action(action, elements)
                print(f"Executed: {result_msg}")
                history.append(f"{action.get('action')}: {result_msg}")
            except Exception as e:
                print(f"Execution error: {e}")
                history.append(f"error: {e}")

            # wait before next screenshot
            time.sleep(SCREENSHOT_DELAY)

        else:
            print(f"\nReached max steps ({MAX_STEPS}), stopping.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        stop_server()


if __name__ == "__main__":
    run("Open Spotlight search and search for Calculator")
