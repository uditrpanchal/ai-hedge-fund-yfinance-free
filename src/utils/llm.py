"""Helper functions for LLM"""

import json
import logging
import re # Ensure re is imported
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory=None,
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # Determine initial response_text
            raw_response_text = "" # For true original logging
            if model_info and not model_info.has_json_mode():
                raw_response_text = result.content if hasattr(result, 'content') else str(result)
            else:
                if isinstance(result, BaseModel):
                    raw_response_text = result.model_dump_json()
                else:
                    raw_response_text = str(result)
            
            # Log the true original raw response from LLM
            # log_message_prefix_original = f"True Original LLM response for agent '{agent_name}'" if agent_name else "True Original LLM response" # REMOVED
            # logging.info(f"{log_message_prefix_original}:\n--BEGIN ORIGINAL RESPONSE--\n{raw_response_text}\n--END ORIGINAL RESPONSE--") # REMOVED

            # Start with raw_response_text for processing, this will be the input to Pydantic
            response_text = raw_response_text 

            if raw_response_text and isinstance(raw_response_text, str):
                # 1. Try to find the overall JSON structure first.
                first_brace_index = response_text.find('{')
                last_brace_index = response_text.rfind('}')

                if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                    json_like_text = response_text[first_brace_index : last_brace_index + 1]
                    
                    try:
                        # Regexes to find individual fields within json_like_text
                        signal_match = re.search(r'"signal"\s*:\s*"([^"]*)"', json_like_text)
                        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_like_text)
                        # This regex for reasoning allows it to span multiple lines due to re.DOTALL
                        # and correctly handles escaped quotes " within the reasoning text.
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:\\"|[^"])*)"', json_like_text, re.DOTALL)

                        if signal_match and confidence_match and reasoning_match:
                            signal_value = signal_match.group(1)
                            confidence_value = float(confidence_match.group(1))
                            reasoning_value = reasoning_match.group(1) # This is a Python string. json.dumps will escape it.
                            
                            reconstructed_data = {
                                "signal": signal_value,
                                "confidence": confidence_value,
                                "reasoning": reasoning_value
                            }
                            response_text = json.dumps(reconstructed_data) # Clean JSON string
                            logging.info(f"Successfully extracted values and reconstructed JSON for agent '{agent_name}'.")
                        else:
                            # Log which fields were not found by regex
                            missing_fields = []
                            if not signal_match: missing_fields.append("signal")
                            if not confidence_match: missing_fields.append("confidence")
                            if not reasoning_match: missing_fields.append("reasoning")
                            logging.warning(f"Regex extraction failed for agent '{agent_name}'. Missing fields: {missing_fields} from block: '{json_like_text[:500]}...'. Falling back.")
                            raise ValueError("Regex extraction failed.") # Trigger fallback

                    except Exception as e_extract: # Catch errors from regex or reconstruction
                        logging.warning(f"Error during regex extraction/reconstruction for agent '{agent_name}': {e_extract}. Falling back to basic cleaning of raw response.")
                        # Fallback logic: basic newline escaping on original raw_response_text
                        # Important: escape backslashes first!
                        response_text = raw_response_text.replace('\\', '\\\\')
                        response_text = response_text.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\r').replace('\"', '\\"')
                        if isinstance(response_text, str): response_text = response_text.strip()
                else: # No { } block found in raw_response_text
                    logging.warning(f"Could not isolate a JSON-like block from raw_response_text for agent '{agent_name}'. Applying basic escapes and strip to raw text.")
                    response_text = raw_response_text.replace('\\', '\\\\')
                    response_text = response_text.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\r').replace('\"', '\\"')
                    if isinstance(response_text, str): response_text = response_text.strip()
            
            # Log the processed response (the one that will be parsed)
            # log_message_prefix_processed = f"Processed LLM response for agent '{agent_name}' (for parsing)" if agent_name else "Processed LLM response (for parsing)" # REMOVED
            # logging.info(f"{log_message_prefix_processed}:\n--BEGIN PROCESSED RESPONSE--\n{response_text}\n--END PROCESSED RESPONSE--") # REMOVED

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                # response_text here is now preprocessed (Markdown stripped, newlines escaped)
                # The extract_json_from_response function expects markdown fences, 
                # so we should call it with the text *before* markdown stripping if it relies on it.
                # However, the goal is to strip markdown *then* parse.
                # So, extract_json_from_response should ideally not be used if we already stripped.
                # Let's assume direct parsing is now possible.
                try:
                    parsed_result = json.loads(response_text)
                    return pydantic_model(**parsed_result)
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError after preprocessing for non-JSON model: {e}. Response text was: {response_text}")
                    raise ValueError(f"Failed to parse preprocessed JSON from non-JSON model response: {e}") from e
                except Exception as e_gen: # Catch other pydantic validation errors etc.
                    logging.error(f"Error constructing pydantic model for non-JSON model after JSON parsing: {e_gen}. Parsed JSON might have been: {response_text[:500]}")
                    raise ValueError(f"Failed to construct pydantic model for non-JSON model: {e_gen}") from e_gen
            else: # This 'else' corresponds to JSON-supporting models (or where with_structured_output is used)
                # If `with_structured_output` was used, `result` should already be the Pydantic model.
                if isinstance(result, pydantic_model):
                    return result # Already parsed by the framework
                
                # If result is not a pydantic model, but we expected JSON mode,
                # it implies `with_structured_output` might not have run or failed.
                # The `response_text` (which is `raw_response_text` processed) should be what we try to parse.
                logging.warning(f"LLM result for JSON mode was not a Pydantic model. Type: {type(result)}. Attempting to parse processed response_text.")
                try:
                    # We use model_validate_json as it's the Pydantic way for raw JSON strings
                    return pydantic_model.model_validate_json(response_text)
                except Exception as parse_error:
                    logging.error(f"Failed to parse LLM response (response_text) into {pydantic_model.__name__} for JSON mode: {parse_error}")
                    raise # Reraise to trigger retry/default

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> Optional[dict]:
    """
    Extracts JSON from markdown-formatted response.
    NOTE: This function might become less relevant if Markdown is stripped *before* parsing
    in the main call_llm logic for non-JSON models. Keeping for now if direct calls are made.
    """
    try:
        # More robust search for ```json block
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            return json.loads(json_text)
        
        # Fallback if no fences but content might be JSON
        if content.strip().startswith("{") and content.strip().endswith("}"):
             return json.loads(content.strip())

    except Exception as e:
        # Using logging instead of print
        logging.error(f"Error extracting JSON from response in extract_json_from_response: {e}. Content was: {content[:500]}")
    return None
