# agent/engine.py
import json
from app.ai.ai_schemas.response import AgentResponse
import os
from app.ai.agent.tools import *
from pathlib import Path
from uuid import UUID, uuid4

MAX_STEPS=10

class JobBoardAIAgent:
     """
        AI Agent for the job board platform.
        - Can call APIs, search embeddings, or use reasoning tools.
        - Maintains context and conversation history for multi-step reasoning.
    """
    def __init__(self, user, session, user_context):
        self.user=user
        self.session=session
        self.context: dict={} # stores intermediate results and tool outputs
        self.messages: list[dict]=[] # conversation history for LLM
        self.user_context=user_context # stores credentials, role, etc.

    @staticmethod
    def get_prompts(category: str):
         """
            Load user or system prompt templates from text files.
        """
        BASE_DIR=Path(__file__).resolve().parent
        user_file_pth=BASE_DIR/"user_prompt.txt"
        system_file_pth=BASE_DIR/"system_prompt.txt"
        if category=="user":
            with open(user_file_pth, "r") as file:
                template=file.read()
            return template
        if category=="system":
            with open(system_file_pth, "r") as file:
                template=file.read()
            return template

    def run(self, task: str):
        """
            Executes the AI agent loop for a given task.
            The agent decides whether to call an API, search embeddings, or reason using LLM.
            Stops if safety triggers or FINAL state is reached.
        """
        USER_PROMPT=self.get_prompts("user")
        SYSTEM_PROMPT=self.get_prompts("system")

        # Initialize conversation with system + user prompts
        self.messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(user_role=self.user.role, context=self.context, task=task)
            }
        ]

        for i in range(MAX_STEPS):
            resp=agent_llm(self.messages)  # Call LLM with current messages
            parsed_resp=AgentResponse.model_validate_json(resp) # Parse response into structured AgentResponse

            if parsed_resp.safety=="not_allowed": # Block unsafe requests
                return {"error": "Unauthorized request"}

            if parsed_resp.state=="FINAL": # Stop if agent declares task finished
                return {
                    "result": parsed_resp.output,
                    "context": self.context
                }

            if parsed_resp.action_type=="api_call": # Handle different actions
                observation=api_call(
                    user=self.user,
                    session=self.session,
                    user_context=self.user_context,
                    action=parsed_resp.action_input
                )
                self.context.setdefault("api_results", []).append(observation)

            elif parsed_resp.action_type=="search_vector_db":
                observation=search_vector_db(
                    session=self.session,
                    query=parsed_resp.action_input.query,
                    entity_type=parsed_resp.action_input.entity_type,
                    top_k=parsed_resp.action_input.top_k
                )
                self.context.setdefault("vector_search_results", []).append(observation)

            elif parsed_resp.action_type=="llm_reasoning_tool": # LLM reasoning may take previous context + task instructions
               observation = llm_reasoning_tool(
                    user_role=self.user.role,
                    task=parsed_resp.action_input.task,
                    context=getattr(parsed_resp.action_input, "context", {}),
                    PROMPT=USER_PROMPT
                )
               self.context["reasoning_output"] = observation

            else:
                return {"error": "Invalid action type"}

             # Save assistant and tool messages in conversation history
            self.messages.append({
                "role": "assistant",
                "content": resp
            })
            self.messages.append({
                "role": "tool",
                "name": parsed_resp.action_type,
                "tool_call_id": str(uuid4()),
                "content": json.dumps(observation)
            })
            # self.messages.append({
            #     "role": "user",
            #     "content": self._build_user_prompt(task)
            # })

        # If max steps exceeded, abort
        return {"error": "Agent exceeded max steps"}
