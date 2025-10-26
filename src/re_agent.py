"""
ReAct Agent for Fitness & Training Coach
Uses LangChain's ReAct framework with exercise lookup tools
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from .tools import exercise_lookup, validate_rest_day, get_current_day


class FitnessReActAgent:
    """ReAct agent for conversational fitness coaching using exercise lookup tools"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.7, enable_tracing: bool = True, project_name: str = "fitness-react-coach"):
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please add it to your .env file."
            )
        
        langsmith_key = os.getenv("LANGSMITH_API_KEY")
        if not langsmith_key:
            raise ValueError(
                "LANGSMITH_API_KEY not found in environment. "
                "Please add it to your .env file."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    def _setup_tools(self) -> List[Tool]:
        """Setup LangChain tools from tools.py"""
        tools = [
            Tool(
                name="exercise_lookup",
                func=exercise_lookup,
                description=(
                    "Search for exercises based on natural language query. "
                    "Use this tool when the user asks about exercises, workouts, or training. "
                    "Input should be a natural language query describing the desired exercises. "
                    "Examples: 'chest exercises with dumbbells', 'best back exercises for strength training', "
                    "'shoulder or chest workouts with high ratings', 'beginner leg exercises at home'. "
                    "Returns formatted exercise recommendations with ratings, equipment, target muscles, and instructions."
                )
            ),
            Tool(
                name="validate_rest_day",
                func=validate_rest_day,
                description=(
                    "IMPORTANT: Check if a weekly workout schedule has too many rest days (max 2 recommended). "
                    "Use this whenever: 1) User provides a schedule, 2) User asks about rest days, "
                    "3) You create a workout plan (ALWAYS validate after), 4) User mentions their training schedule. "
                    "Input format: String with days and activities separated by commas or newlines. "
                    "Example: 'Monday: Chest, Tuesday: Rest, Wednesday: Back, Thursday: Legs, Friday: Rest, Saturday: Arms, Sunday: Rest' "
                    "Returns: Validation result with rest day count and warning if > 2 rest days."
                )
            ),
            Tool(
                name="get_current_day",
                func=get_current_day,
                description=(
                    "Get the current day of the week. "
                    "Use this when creating workout schedules that should start from today. "
                    "No input required. Returns current day name (e.g., 'Monday', 'Tuesday')."
                )
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create ReAct agent with custom prompt"""
        prompt_template = """You are a knowledgeable Fitness & Training Coach assistant. Your role is to help users find exercises, create workout plans, and provide fitness guidance.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
1. When users ask about exercises, workouts, schedules, or training, use the exercise_lookup tool
2. Use get_current_day when creating schedules that should start from today
3. ALWAYS use the validate_rest_day tool when:
   - Users provide or mention a weekly workout schedule
   - Users ask you to create a workout plan (validate it after creation)
   - Users ask about rest days or recovery
   - Users mention their current training schedule
4. For complex requests (e.g., workout plans), break them down into multiple tool calls
5. After creating any weekly workout plan, automatically validate it with validate_rest_day
6. Provide helpful context and explanations with exercise recommendations
7. If users ask about safety or medical concerns, remind them to consult healthcare professionals
8. Be encouraging and supportive in your responses
Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def query(self, user_input: str) -> str:
        """
        Process user query through ReAct agent
        
        Args:
            user_input: User's question or request
            
        Returns:
            Agent's response with exercise recommendations and guidance
        """
        try:
            result = self.agent_executor.invoke(
                {"input": user_input},
                config={
                    "metadata": {
                        "user_input_len": len(user_input),
                        "model": getattr(self.llm, "model", "gemini"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
            return result["output"]
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg
    
    def chat(self):
        """Interactive chat interface for the ReAct agent"""
        print("="*60)
        print("FITNESS & TRAINING COACH - ReAct Agent")
        print("="*60)
        print("\nI'm your AI fitness coach! I can help you:")
        print("  • Find exercises for specific muscle groups")
        print("  • Recommend workouts based on equipment")
        print("  • Suggest exercises for different fitness levels")
        print("  • Create custom workout routines")
        print("  • Validate your weekly schedule for proper rest days")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("="*60)
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThanks for using Fitness Coach! Stay active and healthy! 💪")
                break
            
            if not user_input:
                continue
            
            conversation_history.append({"role": "user", "content": user_input})
            
            print("\nCoach: ", end="", flush=True)
            try:
                response = self.query(user_input)
                print(response)
                conversation_history.append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")


def main():
    """Main entry point for ReAct agent"""
    print("Initializing Fitness Coach ReAct Agent...")
    
    try:
        agent = FitnessReActAgent()
        print("✓ Agent initialized successfully!\n")
        
        agent.chat()
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure:")
        print("  1. You have a .env file in the project root")
        print("  2. GEMINI_API_KEY is set in the .env file")
        print("  3. The vector database is built (run 'make index')")
        return
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (run 'make install-mac' or 'make install')")
        print("  2. The vector database exists (run 'make index')")
        return


if __name__ == "__main__":
    main()
