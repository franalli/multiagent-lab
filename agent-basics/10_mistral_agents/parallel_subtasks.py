import asyncio
import json
import os

import nest_asyncio
import requests
from IPython.display import Markdown, display
from mistralai import Mistral
from pydantic import BaseModel

nest_asyncio.apply()

MISTRAL_API_KEY = os.environ.get(
    "MISTRAL_API_KEY", "<YOUR MISTRAL API KEY>"
)  # Get it from https://console.mistral.ai/api-keys
TAVILY_API_KEY = os.environ.get(
    "TAVILY_API_KEY", "<YOUR TAVILY API KEY>"
)  # Get it from https://app.tavily.com/home

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-small-latest"  # Can be configured based on needs

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_HEADERS = {
    "Authorization": f"Bearer {TAVILY_API_KEY}",
    "Content-Type": "application/json",
}


class SubTask(BaseModel):
    """Individual subtask definition"""

    task_id: str
    type: str
    description: str
    search_query: str | None  # Query for Tavily search for the subtask


class TaskList(BaseModel):
    """Structure for orchestrator output"""

    analysis: str
    subtasks: list[SubTask]


def fetch_information(query: str, max_results: int = 3):
    """Retrieve information from Tavily API"""
    payload = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": max_results,
    }

    try:
        response = requests.post(TAVILY_API_URL, json=payload, headers=TAVILY_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Tavily: {e}")
        return {"error": str(e), "results": []}


def run_mistral_llm(prompt: str, system_prompt: str | None = None):
    """Run Mistral LLM with given prompts"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = mistral_client.chat.complete(
        model=MISTRAL_MODEL, messages=messages, temperature=0.7, max_tokens=4000
    )

    return response.choices[0].message.content


def parse_structured_output(
    prompt: str, response_format: BaseModel, system_prompt: str | None = None
):
    """Get structured output from Mistral LLM based on a Pydantic model"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = mistral_client.chat.parse(
        model=MISTRAL_MODEL,
        messages=messages,
        response_format=response_format,
        temperature=0.2,
    )

    return json.loads(response.choices[0].message.content)


async def run_task_async(task: SubTask, original_task: str):
    """Execute a single subtask asynchronously with Tavily enhancement"""

    # Prepare context with Tavily information if a search query is provided
    context = ""
    if task.search_query:
        print(f"Fetching information for: {task.search_query}")
        search_results = fetch_information(task.search_query)

        # Format search results into context
        if search_results.get("results"):
            context = "### Relevant Information:\n"
            for result in search_results["results"]:
                context += f"- {result.get('content', '')}\n"

        if search_results.get("answer"):
            context += f"\n### Summary: {search_results['answer']}\n"

    # Worker prompt with task information and context
    worker_prompt = f"""
    Complete the following subtask based on the given information:

    Original Task: {original_task}

    Subtask Type: {task.type}
    Subtask Description: {task.description}

    {context}

    Please provide a detailed response for this specific subtask only.
    """

    # Use asyncio to run in a thread pool to prevent blocking
    return await asyncio.to_thread(
        run_mistral_llm,
        prompt=worker_prompt,
        system_prompt="You are a specialized agent focused on solving a specific aspect of a larger task.",
    )


async def execute_tasks_in_parallel(subtasks: list[SubTask], original_task: str):
    """Execute all subtasks in parallel"""
    tasks = []
    for subtask in subtasks:
        tasks.append(run_task_async(subtask, original_task))

    return await asyncio.gather(*tasks)


async def workflow(user_task: str):
    """Main workflow function to process a task through the parallel subtask agent workflow"""

    print("=== USER TASK ===\n")
    print(user_task)

    # Step 1: Orchestrator breaks down the task into subtasks
    orchestrator_prompt = f"""
    Analyze this task and break it down into 3-5 distinct, specialized subtasks that could be executed in parallel:

    Task: {user_task}

    For each subtask:
    1. Assign a unique task_id
    2. Define a specific type that describes the subtask's focus
    3. Write a clear description explaining what needs to be done
    4. Provide a search query if the subtask requires additional information

    First, provide a brief analysis of your understanding of the task.
    Then, define the subtasks that would collectively solve this problem effectively.

    Remember to make the subtasks complementary, not redundant, focusing on different aspects of the problem.
    """

    orchestrator_system_prompt = """
    You are a task orchestrator that specializes in breaking down complex problems into smaller,
    well-defined subtasks that can be solved independently and in parallel. Think carefully about
    the most logical way to decompose the given task.
    """

    print("\nOrchestrating task decomposition...")
    # Get structured output from orchestrator
    task_breakdown = parse_structured_output(
        prompt=orchestrator_prompt,
        response_format=TaskList,
        system_prompt=orchestrator_system_prompt,
    )

    # Display orchestrator output
    print("\n=== ORCHESTRATOR OUTPUT ===")
    print(f"\nANALYSIS:\n{task_breakdown['analysis']}")
    print("\nSUBTASKS:")
    for task in task_breakdown["subtasks"]:
        print(f"- {task['task_id']}: {task['type']} - {task['description'][:100]}...")

    # Step 2: Execute subtasks in parallel
    print("\nExecuting subtasks in parallel...")
    subtask_results = await execute_tasks_in_parallel(
        [SubTask(**task) for task in task_breakdown["subtasks"]], user_task
    )

    # Display worker results
    for i, (task, result) in enumerate(
        zip(task_breakdown["subtasks"], subtask_results)
    ):
        print(f"\n=== WORKER RESULT ({task['type']}) ===")
        print(f"{result[:200]}...\n")

    # Step 3: Synthesize final response
    print("\nSynthesizing final response...")

    # Format worker responses for synthesizer
    worker_responses = ""
    for task, response in zip(task_breakdown["subtasks"], subtask_results):
        worker_responses += f"\n=== SUBTASK: {task['type']} ===\n{response}\n"

    synthesizer_prompt = f"""
    Given the following task: {user_task}

    And these responses from different specialized agents focusing on different aspects of the task:

    {worker_responses}

    Please synthesize a comprehensive, coherent response that addresses the original task.
    Integrate insights from all specialized agents while avoiding redundancy.
    Ensure your response is balanced, considering all the different perspectives provided.
    """

    final_response = run_mistral_llm(
        prompt=synthesizer_prompt,
        system_prompt="You are a synthesis agent that combines specialized analyses into comprehensive responses.",
    )

    return {
        "orchestrator_analysis": task_breakdown["analysis"],
        "subtasks": task_breakdown["subtasks"],
        "subtask_results": subtask_results,
        "final_response": final_response,
    }


# Examples
task = "Compare the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro, and recommend which one I should purchase."
result = asyncio.run(workflow(task))

print("\n=== FINAL SYNTHESIZED RESPONSE ===")
display(Markdown(result["final_response"]))
