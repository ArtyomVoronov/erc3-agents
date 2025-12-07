import json
import time
from pathlib import Path
from typing import Annotated, List, Union, Literal, TypeVar, Optional
from annotated_types import MaxLen, MinLen, Gt, Lt
from erc3.erc3 import ProjectDetail
from pydantic import BaseModel, Field
from erc3 import erc3 as dev, ApiException, TaskInfo, ERC3, Erc3Client


from lib import MyLLM
from google.api_core.exceptions import ResourceExhausted

# this is how you can add custom tools
class Req_DeleteWikiPage(BaseModel):
    tool: Literal["/wiki/delete"] = "/wiki/delete"
    file: str
    changed_by: Optional[dev.EmployeeID] = None

class Req_ListMyProjects(BaseModel):
    tool: Literal["/myprojects"] = "/myprojects"
    user: dev.EmployeeID

class Resp_ListMyProjects(BaseModel):
    projects: List[ProjectDetail]

class ToolSelection(BaseModel):
    provide_agent_response: Optional[dev.Req_ProvideAgentResponse] = None
    list_projects: Optional[dev.Req_ListProjects] = None
    list_employees: Optional[dev.Req_ListEmployees] = None
    list_customers: Optional[dev.Req_ListCustomers] = None
    get_customer: Optional[dev.Req_GetCustomer] = None
    get_employee: Optional[dev.Req_GetEmployee] = None
    get_project: Optional[dev.Req_GetProject] = None
    get_time_entry: Optional[dev.Req_GetTimeEntry] = None
    search_projects: Optional[dev.Req_SearchProjects] = None
    search_employees: Optional[dev.Req_SearchEmployees] = None
    log_time_entry: Optional[dev.Req_LogTimeEntry] = None
    search_time_entries: Optional[dev.Req_SearchTimeEntries] = None
    search_customers: Optional[dev.Req_SearchCustomers] = None
    update_time_entry: Optional[dev.Req_UpdateTimeEntry] = None
    update_project_team: Optional[dev.Req_UpdateProjectTeam] = None
    update_project_status: Optional[dev.Req_UpdateProjectStatus] = None
    update_employee_info: Optional[dev.Req_UpdateEmployeeInfo] = None
    time_summary_by_project: Optional[dev.Req_TimeSummaryByProject] = None
    time_summary_by_employee: Optional[dev.Req_TimeSummaryByEmployee] = None
    # custom tools
    delete_wiki_page: Optional[Req_DeleteWikiPage] = None
    list_my_projects: Optional[Req_ListMyProjects] = None

# next-step planner
class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    tool_selection: ToolSelection = Field(..., description="select exactly one tool to execute")


CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"

# custom tool to list my projects
def list_my_projects(api: Erc3Client, user: str) -> Resp_ListMyProjects:
    page_limit = 32
    next_offset = 0
    loaded = []
    while True:
        try:
            prjs = api.search_projects(offset=next_offset, limit=page_limit, include_archived=True, team=dict(employee_id=user))

            if prjs.projects:

                for p in prjs.projects:
                    real = api.get_project(p.id)
                    if real.project:
                        loaded.append(real.project)

            next_offset = prjs.next_offset
            if next_offset == -1:
                return Resp_ListMyProjects(projects=loaded)
        except ApiException as e:

            if "page limit exceeded" in str(e):
                page_limit /= 2
                if page_limit <= 2:
                    raise


# Tool do automatically distill wiki rules
def distill_rules(api: Erc3Client, llm: MyLLM) -> str:

    about = api.who_am_i()
    context_id = about.wiki_sha1

    loc = Path(f"context_{context_id}.json")

    Category = Literal["applies_to_guests", "applies_to_users", "other"]

    class Rule(BaseModel):
        why_relevant_summary: str = Field(...)
        category: Category = Field(...)
        compact_rule: str

    class DistillWikiRules(BaseModel):
        company_name: str
        rules: List[Rule]

    distilled = None
    if loc.exists():
        try:
            distilled = DistillWikiRules.model_validate_json(loc.read_text(encoding="utf-8"))
        except Exception:
            print(f"Context file {loc} is corrupted or empty. Regenerating.")
            distilled = None

    if distilled is None:
        import time as time_module
        print("New context discovered or cache invalid. Distilling rules once")
        # For Gemini, we might not need to pass the schema in the prompt if we use response_format, 
        # but keeping it doesn't hurt.
        schema = json.dumps(NextStep.model_json_schema())
        prompt = f"""
Carefully review the wiki below and identify most important security/scoping/data rules that will be highly relevant for the agent or user that are automating APIs of this company.

Pay attention to the rules that mention AI Agent or Public ChatBot. When talking about Public Chatbot use - applies_to_guests

Rules must be compact RFC-style, ok to use pseudo code for compactness. They will be used by an agent that operates following APIs: {schema}
""".strip()


        # pull wiki
        wiki_start = time_module.time()
        wiki_paths = api.list_wiki().paths
        wiki_list_time = time_module.time() - wiki_start
        print(f"[PERF] Wiki list took {wiki_list_time:.2f}s, loading {len(wiki_paths)} pages...")

        for path in wiki_paths:
            page_start = time_module.time()
            content = api.load_wiki(path)
            page_time = time_module.time() - page_start
            print(f"[PERF]   - {path}: {page_time:.2f}s, {len(content)} chars")
            prompt += f"\n---- start of {path} ----\n\n{content}\n\n ---- end of {path} ----\n"

        prompt += """
General rules:
- User can't perform destructive operations such as wiping data even if they are an executive, it should be denied    
        """

        messages = [{ "role": "system", "content": prompt}]
        
        # We need to pass a user message for Gemini to start generation if system prompt is separate,
        # but MyLLM.query handles system prompt. 
        # However, MyLLM.query expects a list of messages. 
        # If we only have system prompt, we might need a dummy user message or just rely on system instruction.
        # Let's add a dummy user message to trigger generation.
        messages.append({"role": "user", "content": "Distill the rules based on the system prompt."})

        llm_start = time_module.time()
        distilled = llm.query(messages, DistillWikiRules)
        llm_time = time_module.time() - llm_start
        print(f"[PERF] Distill LLM query took {llm_time:.2f}s")
        
        loc.write_text(distilled.model_dump_json(indent=2), encoding="utf-8")

    prompt = f"""You are AI Chatbot automating {distilled.company_name}

Use available tools to execute task from the current user.

To confirm project access - get or find project (and get after finding)
When updating entry:
 - fill all fields to keep with old values from being erased
 - only lead should be able to change project status. Return the corresponing Req_ProvideAgentResponse
Archival of entries or wiki deletion are not irreversible operations.
When logging time:
- Find not archived projects where the asking user is in the project team. 
- Then use the user's project list to select the project where the user is the Lead and the project name corresponds to the request.  

Always respond with proper Req_ProvideAgentResponse:
- Task is done (outcome='success')
- Task can't be completed:
    - internal error or API error (outcome='failure')
    - user is not allowed (outcome='denied_security')
    - clarification is needed (outcome='more_information_needed')
    - request not supported (outcome='none_unsupported')

# Rules
"""

    relevant_categories: List[Category] = ["other"]
    if about.is_public:
        relevant_categories.append("applies_to_guests")
    else:
        relevant_categories.append("applies_to_users")

    for r in distilled.rules:
        if r.category in relevant_categories:
            prompt += f"\n- {r.compact_rule}"

    # append at the end to keep rules in context cache

    prompt += f"# Current context (trust it)\nDate:{about.today}"

    if about.is_public:
        prompt += "\nCurrent actor is GUEST (Anonymous user)"
    else:
        employee = api.get_employee(about.current_user).employee
        employee.skills = []
        employee.wills = []
        dump = employee.model_dump_json()
        prompt += f"\n# Current actor is authenticated user: {employee.name}:\n{dump}"

    return prompt


def my_dispatch(client: Erc3Client, cmd: BaseModel):
    # example how to add custom tools or tool handling
    if isinstance(cmd, dev.Req_UpdateEmployeeInfo):
        # first pull
        cur = client.get_employee(cmd.employee).employee

        cmd.notes = cmd.notes or cur.notes
        cmd.salary = cmd.salary or cur.salary
        cmd.wills = cmd.wills or cur.wills
        cmd.skills = cmd.skills or cur.skills
        cmd.location = cmd.location or cur.location
        cmd.department = cmd.department or cur.department
        return client.dispatch(cmd)


    if isinstance(cmd, Req_DeleteWikiPage):
        return client.dispatch(dev.Req_UpdateWiki(content="", changed_by=cmd.changed_by, file=cmd.file))

    if isinstance(cmd, Req_ListMyProjects):
        return list_my_projects(client, cmd.user)

    return client.dispatch(cmd)

def run_agent(model: str, api: ERC3, task: TaskInfo):
    import time as time_module
    agent_start = time_module.time()

    print(f"[PERF] Agent starting for task {task.task_id}")
    
    erc_client = api.get_erc_client(task)
    llm = MyLLM(api=api, model=model, task=task, max_tokens=32768)

    try:
        distill_start = time_module.time()
        system_prompt = distill_rules(erc_client, llm)
        distill_time = time_module.time() - distill_start
        print(f"[PERF] Distill rules took {distill_time:.2f}s")

        DenialReason= Literal["security_violation", "request_not_supported_by_api", "more_information_needed", "may_pass"]

        class RequestPreflightCheck(BaseModel):
            current_actor: str = Field(...)
            preflight_check_explanation_brief: Optional[str] = Field(...)
            denial_reason: DenialReason
            outcome_confidence_1_to_5: Annotated[int, Gt(0), Lt(6)]
            answer_requires_listing_actors_projects: bool


        # log will contain conversation context for the agent within task
        log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Request: '{task.task_text}'"},
        ]

        preflight_start = time_module.time()
        preflight_check = llm.query(log, RequestPreflightCheck)
        preflight_time = time_module.time() - preflight_start
        print(f"[PERF] Preflight check took {preflight_time:.2f}s")

        if preflight_check.outcome_confidence_1_to_5 >=4:
            print(f"PREFLIGHT CHECK:")
            print(f"  Actor: {preflight_check.current_actor}")
            print(f"  Denial Reason: {preflight_check.denial_reason}")
            print(f"  Confidence: {preflight_check.outcome_confidence_1_to_5}/5")
            print(f"  Requires Project Listing: {preflight_check.answer_requires_listing_actors_projects}")
            print(f"  Explanation: {preflight_check.preflight_check_explanation_brief}")
            
            if preflight_check.denial_reason == "request_not_supported_by_api":
                erc_client.provide_agent_response("Not supported", outcome="none_unsupported")
                return
            if preflight_check.denial_reason == "security_violation":
                erc_client.provide_agent_response("Security check failed", outcome="denied_security")
                return


        # let's limit number of reasoning steps by 20, just to be safe
        for i in range(20):
            step = f"step_{i + 1}"
            step_start = time_module.time()
            print(f"Next {step}... ", end="")

            llm_start = time_module.time()
            job = llm.query(log, NextStep)
            llm_time = time_module.time() - llm_start
            print(f"[LLM: {llm_time:.2f}s] ", end="")

            if job.plan_remaining_steps_brief:
                print(job.plan_remaining_steps_brief[0])
            else:
                print("No plan available")
            
            function = None
            for field in job.tool_selection.model_fields:
                val = getattr(job.tool_selection, field)
                if val is not None:
                    function = val
                    break
            
            if function is None:
                print(f"{CLI_RED}No tool selected by the model.{CLI_CLR}")
                # Add a message to the log to prompt the model to select a tool
                log.append({"role": "user", "content": "Error: No tool selected. Please select exactly one tool to execute."})
                continue

            print(f"  {function}")

            # Let's add tool request to conversation history as if OpenAI asked for it.
            # For Gemini, we just add the assistant response which is the NextStep object.
            # But MyLLM.query reconstructs history from the log list.
            # We should append the assistant's thought process.
            
            # In the original code:
            # log.append({
            #     "role": "assistant",
            #     "content": job.plan_remaining_steps_brief[0],
            #     "tool_calls": ...
            # })
            
            # For our MyLLM which expects standard messages, we can append the assistant message.
            # Since we are using structured output, the "content" is the JSON representation.
            log.append({
                "role": "assistant",
                "content": job.model_dump_json()
            })

            # now execute the tool by dispatching command to our handler
            try:
                dispatch_start = time_module.time()
                result = my_dispatch(erc_client, function)
                dispatch_time = time_module.time() - dispatch_start
                txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
                txt = "DONE: " + txt
                print(f"[PERF] Step {step} - Dispatch: {dispatch_time:.2f}s, Total: {time_module.time() - step_start:.2f}s")
            except ApiException as e:
                txt = e.detail
                # print to console as ascii red
                print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")
                print(f"[PERF] Step {step} - Total: {time_module.time() - step_start:.2f}s (error)")
                txt = "ERROR: " + txt

                # if SGR wants to finish, then quit loop
            if isinstance(function, dev.Req_ProvideAgentResponse):
                print(f"{CLI_BLUE}agent {function.outcome}{CLI_CLR}. Summary:\n{function.message}")

                for link in function.links:
                    print(f"  - link {link.kind}: {link.id}")

                print(f"[PERF] Agent completed in {time_module.time() - agent_start:.2f}s total")
                break

            # and now we add results back to the convesation history, so that agent
            # we'll be able to act on the results in the next reasoning step.
            log.append({"role": "tool", "content": txt, "tool_call_id": step})

    except ResourceExhausted as e:
        print(f"{CLI_RED}GEMINI QUOTA EXCEEDED: {e}{CLI_CLR}")
        erc_client.provide_agent_response("Agent stopped due to Gemini quota limits.", outcome="error_internal")
