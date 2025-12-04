import argparse
import textwrap
from agent import run_agent
from erc3 import ERC3

MODEL_ID = "gemini-3-pro-preview"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini Agent")
    parser.add_argument("--task", type=str, help="Run a specific task by spec ID (e.g. project_check_by_member)")
    args = parser.parse_args()

    core = ERC3()

    if args.task:
        # Debugging a single task
        print(f"Running single task: {args.task}")
        task = core.start_new_task("erc3-test", args.task)
        with open("debug_task.txt", "w") as f:
            f.write(f"Type: {type(task)}\n")
            f.write(f"Dir: {dir(task)}\n")
            f.write(f"Content: {task}\n")
        
        run_agent(MODEL_ID, core, task)
    else:
        # Start session with metadata
        res = core.start_session(
            benchmark="erc3-test",
            workspace="my",
            name=f"NextStep SGR ({MODEL_ID}) @Artem",
            architecture="NextStep SGR Agent with Gemini")

        status = core.session_status(res.session_id)
        print(f"Session has {len(status.tasks)} tasks")

        for task in status.tasks:
            print("="*40)
            print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
            core.start_task(task)
            try:
                run_agent(MODEL_ID, core, task)
            except Exception as e:
                print(e)
            result = core.complete_task(task)
            if result.eval:
                explain = textwrap.indent(result.eval.logs, "  ")
                print(f"\nSCORE: {result.eval.score}\n{explain}\n")

        core.submit_session(res.session_id)













