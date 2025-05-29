import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
import os # Added for os.getenv
import re # For JSON parsing

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.utils.visualize import save_graph_as_png
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response_input):
    """Parses a JSON string (potentially embedded in markdown) and returns a dictionary."""
    response_to_parse = response_input
    original_response_for_error = response_input # Keep original for full error context if needed

    if isinstance(response_to_parse, str):
        # Try to extract content within ```json ... ```
        match = re.search(r"```json\s*(.*?)\s*```", response_to_parse, re.DOTALL | re.IGNORECASE)
        if match:
            response_to_parse = match.group(1)
        else:
            # If no ```json ... ```, try to find content within ``` ... ``` (any language or no language)
            match = re.search(r"```\s*(.*?)\s*```", response_to_parse, re.DOTALL | re.IGNORECASE)
            if match:
                response_to_parse = match.group(1)
            else:
                # If no backticks, look for the first '{' and last '}' or first '[' and last ']'
                # This is a simpler heuristic. More robust might be matching balanced braces/brackets.
                # For now, using greedy matching up to the last corresponding symbol.
                brace_match = re.search(r"(\{.*\})", response_to_parse, re.DOTALL)
                bracket_match = re.search(r"(\[.*\])", response_to_parse, re.DOTALL) # Catches arrays of objects

                # Prioritize object match, then array match
                if brace_match and bracket_match:
                    # If both object and array are found, choose the one that starts earlier
                    if brace_match.start() <= bracket_match.start():
                        response_to_parse = brace_match.group(1)
                    else:
                        response_to_parse = bracket_match.group(1)
                elif brace_match:
                    response_to_parse = brace_match.group(1)
                elif bracket_match:
                    response_to_parse = bracket_match.group(1)
                # If neither, response_to_parse remains the original string (if it was a string)

    try:
        return json.loads(response_to_parse)
    except json.JSONDecodeError as e:
        # Print the string that was attempted for parsing, and also the original if different
        error_context = f"Attempted to parse: {repr(response_to_parse)}"
        if response_to_parse is not original_response_for_error:
            error_context += f"\nOriginal response: {repr(original_response_for_error)}"
        print(f"JSON decoding error: {e}\n{error_context}")
        return None
    except TypeError as e: # Handles if response_to_parse is not string-like (e.g. if input was not string and regex failed)
        print(f"Invalid response type for JSON parsing (expected string, got {type(response_to_parse).__name__}): {e}\nOriginal input: {repr(original_response_for_error)}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nAttempted to parse: {repr(response_to_parse)}\nOriginal response: {repr(original_response_for_error)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position. Defaults to 100000.0)")
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement. Defaults to 0.0")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference (interactive only if LLM args not set)")
    parser.add_argument("--analysts", type=str, help="Comma-separated list of analyst keys (e.g., aswath_damodaran,ben_graham)")
    parser.add_argument("--llm-model-name", type=str, help="Name of the LLM model to use (e.g., gpt-4o)")
    parser.add_argument("--llm-model-provider", type=str, help="Provider of the LLM model (e.g., OpenAI, Ollama)")

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # --- Analyst Selection ---
    selected_analysts = []
    if args.analysts:
        provided_analyst_keys = [key.strip() for key in args.analysts.split(",")]
        valid_analyst_map = {value: display for display, value in ANALYST_ORDER} # For validation and display name lookup
        
        invalid_keys = [key for key in provided_analyst_keys if key not in valid_analyst_map]
        if invalid_keys:
            print(f"{Fore.RED}Error: Invalid analyst key(s) provided: {', '.join(invalid_keys)}{Style.RESET_ALL}")
            print(f"Valid keys are: {', '.join(valid_analyst_map.keys())}")
            sys.exit(1)
        
        selected_analysts = provided_analyst_keys
        analyst_display_names = [valid_analyst_map.get(key, key).title().replace('_', ' ') for key in selected_analysts]
        print(f"Using command-line selected analysts: {', '.join(Fore.GREEN + name + Style.RESET_ALL for name in analyst_display_names)}\n")
    else:
        choices_interactive = questionary.checkbox(
            "Select your AI analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                    [
                        ("checkbox-selected", "fg:green"),
                        ("selected", "fg:green noinherit"),
                        ("highlighted", "noinherit"),
                        ("pointer", "noinherit"),
                    ]
                ),
            ).ask()

        if not choices_interactive:
            print("\n\nInterrupt received or no selection made. Exiting...")
            sys.exit(0)
        else:
            selected_analysts = choices_interactive
            # Get display names for printing
            valid_analyst_map = {value: display for display, value in ANALYST_ORDER}
            analyst_display_names = [valid_analyst_map.get(key, key).title().replace('_', ' ') for key in selected_analysts]
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + name + Style.RESET_ALL for name in analyst_display_names)}\n")

    # --- LLM Model Selection ---
    model_name = ""
    model_provider = ""

    if args.llm_model_name and args.llm_model_provider:
        model_name = args.llm_model_name
        model_provider = args.llm_model_provider
        
        # Validate provider string against ModelProvider enum
        try:
            # Case-insensitive check for provider by trying to match it with ModelProvider values
            matched_provider = False
            for enum_member in ModelProvider:
                if model_provider.lower() == enum_member.value.lower():
                    model_provider = enum_member.value # Use the canonical casing
                    matched_provider = True
                    break
            if not matched_provider:
                raise ValueError(f"Invalid LLM provider: {args.llm_model_provider}")
        except ValueError:
            valid_providers_str = ", ".join([mp.value for mp in ModelProvider])
            print(f"{Fore.RED}Error: Invalid LLM model provider '{args.llm_model_provider}'. Valid providers are: {valid_providers_str}{Style.RESET_ALL}")
            sys.exit(1)

        print(f"Using command-line selected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

        if model_provider == ModelProvider.OLLAMA.value: # Check against canonical value
            if not ensure_ollama_and_model(model_name):
                print(f"{Fore.RED}Ollama setup issue: Cannot proceed with model '{model_name}'.{Style.RESET_ALL}")
                sys.exit(1)
            # If provider is Ollama via args, set args.ollama to True for any downstream logic that might use it,
            # though ensure_ollama_and_model already did the main check.
            args.ollama = True # This ensures that if other parts of code check args.ollama, it's consistent
    
    elif args.llm_model_name or args.llm_model_provider: # Only one is provided
        print(f"{Fore.YELLOW}Warning: Both --llm-model-name and --llm-model-provider must be specified for non-interactive LLM selection.{Style.RESET_ALL}")
        print("Falling back to interactive LLM selection...\n")
        # Fall through to interactive selection below, args.ollama will be False unless explicitly set by user

    if not model_name or not model_provider: # Fallback to interactive if not fully specified by args
        # Use args.ollama to guide interactive prompt type
        if args.ollama: 
            print(f"{Fore.CYAN}Using Ollama for local LLM inference (interactive selection).{Style.RESET_ALL}")
            interactive_model_name = questionary.select(
                "Select your Ollama model:",
                choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
                style=questionary.Style(
                    [("selected", "fg:green bold"), ("pointer", "fg:green bold"), ("highlighted", "fg:green"), ("answer", "fg:green bold")]
                ),
            ).ask()

            if not interactive_model_name: print("\n\nInterrupt received. Exiting..."); sys.exit(0)
            if interactive_model_name == "-": # Custom model
                interactive_model_name = questionary.text("Enter the custom model name:").ask()
                if not interactive_model_name: print("\n\nInterrupt received. Exiting..."); sys.exit(0)
            
            model_name = interactive_model_name # Set main model_name
            if not ensure_ollama_and_model(model_name): # Validate selected/custom Ollama model
                print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}"); sys.exit(1)
            model_provider = ModelProvider.OLLAMA.value # Set main model_provider
            print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        
        else: # Standard cloud/API model interactive selection (args.ollama is False)
            model_choice_interactive = questionary.select(
                    "Select your LLM model:",
                    choices=[questionary.Choice(display, value=(name, provider_val)) for display, name, provider_val in LLM_ORDER],
                    style=questionary.Style(
                        [("selected", "fg:green bold"), ("pointer", "fg:green bold"), ("highlighted", "fg:green"), ("answer", "fg:green bold")]
                    ),
                ).ask()

            if not model_choice_interactive: print("\n\nInterrupt received. Exiting..."); sys.exit(0)
            
            interactive_model_name, interactive_model_provider = model_choice_interactive
            model_info = get_model_info(interactive_model_name, interactive_model_provider)

            if model_info and model_info.is_custom():
                interactive_model_name_custom = questionary.text("Enter the custom model name:").ask()
                if not interactive_model_name_custom: print("\n\nInterrupt received. Exiting..."); sys.exit(0)
                model_name = interactive_model_name_custom
            else:
                model_name = interactive_model_name # Set main model_name
            
            model_provider = interactive_model_provider # Set main model_provider
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    if not selected_analysts:
        print("\nNo analysts selected. Exiting.") 
        sys.exit(0)
        
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": args.initial_cash,  # Initial cash amount
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        "margin_used": 0.0,  # total margin usage across all short positions
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
                "short_margin_used": 0.0,  # Dollars of margin used for this ticker's short
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            }
            for ticker in tickers
        },
    }

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)
