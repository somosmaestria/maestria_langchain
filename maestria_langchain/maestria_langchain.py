import requests
from typing import Any, Dict, List, Optional, Type, Union, Literal
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from pydantic import BaseModel, Field
import json
import ast


class Classification(BaseModel):
    """
    Schema for extracting the sentiment of an email
    The sentiment can be positive, negative, or neutral.
    """

    country: str = Field(description="return the country of the email")
    country_state: str = Field(description="return the state of the email")


class MaestriaLLM(LLM):
    model: str = "deepseek-r1:8b"
    api_key: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = False
    system_message: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "maestria-llms"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        method: Literal["json_mode"] = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be a Pydantic class or a dict schema.
            method: The method for steering model generation (only "json_mode" supported).
            include_raw: If True, returns both raw and parsed output.
            **kwargs: Additional keyword arguments.

        Returns:
            A Runnable that takes string inputs and returns structured output.
        """

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser = PydanticOutputParser(pydantic_object=schema)
            schema_dict = schema.model_json_schema()
            schema_name = schema.__name__
        else:
            parser = JsonOutputParser()
            schema_dict = schema
            schema_name = "Output"

        class StructuredOutputRunnable(Runnable):
            def __init__(
                self, llm: MaestriaLLM, parser, schema_dict: Dict, include_raw: bool
            ):
                self.llm = llm
                self.parser = parser
                self.schema_dict = schema_dict
                self.include_raw = include_raw

            def invoke(
                self, input_data: Union[str, Dict], **kwargs
            ) -> Union[Dict, BaseModel]:
                # Extract the actual prompt from input
                if isinstance(input_data, str):
                    prompt = input_data
                elif isinstance(input_data, dict):
                    prompt = input_data.get("input", str(input_data))
                else:
                    prompt = str(input_data)

                # Create a structured prompt
                structured_prompt = self._create_structured_prompt(prompt)

                # Get raw response from LLM
                raw_response = self.llm._call(structured_prompt)

                if self.include_raw:
                    result = {"raw": raw_response}
                    try:
                        parsed = self._parse_response(raw_response)
                        result["parsed"] = parsed
                        result["parsing_error"] = None
                    except Exception as e:
                        result["parsed"] = None
                        result["parsing_error"] = e
                    return result
                else:
                    return self._parse_response(raw_response)

            def _create_structured_prompt(self, prompt: str) -> str:
                """Create a prompt that encourages structured JSON output."""
                schema_str = json.dumps(self.schema_dict, indent=2)

                return f"""You must respond with valid JSON that matches this exact schema:

{schema_str}

Instructions:
- Return only valid JSON, no additional text
- Include all required fields from the schema
- Follow the field descriptions carefully

User input: {prompt}

JSON response:"""

            def _parse_response(self, response: str) -> Union[Dict, BaseModel]:
                """Parse the LLM response into structured output."""
                # Ensure response is a string
                if not isinstance(response, str):
                    response = str(response)
                    
                if response.startswith("Error:") or response.startswith("Request failed:"):
                    raise ValueError(f"LLM error: {response}")
                
                # First, try to parse as direct JSON
                cleaned_response = self._clean_json_response(response)
                
                try:
                    # Try to parse as direct JSON first
                    json_data = json.loads(cleaned_response)
                    
                    # If using Pydantic, validate and return instance
                    if isinstance(self.parser, PydanticOutputParser):
                        return self.parser.parse(json.dumps(json_data))
                    else:
                        return json_data
                        
                except json.JSONDecodeError:
                    # If that fails, the response might be a string representation of a dict
                    # This happens when the API returns the full response structure as a string
                    try:
                        # Safely evaluate the string as a Python literal
                        api_response = ast.literal_eval(response)
                        
                        if isinstance(api_response, dict) and "response" in api_response:
                            actual_content = api_response["response"]
                            # Parse the actual JSON content
                            json_data = json.loads(actual_content)
                            
                            if isinstance(self.parser, PydanticOutputParser):
                                return self.parser.parse(json.dumps(json_data))
                            else:
                                return json_data
                        else:
                            raise ValueError(f"Unexpected response format: {response}")
                            
                    except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                        # Last resort: try to extract JSON from the string manually
                        try:
                            # Look for the 'response': 'JSON_CONTENT' pattern
                            import re
                            pattern = r"'response':\s*'([^']+)'"
                            match = re.search(pattern, response)
                            if match:
                                json_content = match.group(1)
                                # Unescape the JSON content
                                json_content = json_content.replace('\\n', '\n').replace('\\"', '"')
                                json_data = json.loads(json_content)
                                
                                if isinstance(self.parser, PydanticOutputParser):
                                    return self.parser.parse(json.dumps(json_data))
                                else:
                                    return json_data
                            else:
                                raise ValueError(f"Could not extract JSON from response: {response}")
                        except Exception as final_e:
                            raise ValueError(f"Could not parse response after all attempts: {response}") from final_e
                        
                except Exception as e:
                    raise ValueError(f"Error parsing structured output: {str(e)}") from e

            def _clean_json_response(self, response: str) -> str:
                """Clean the response to extract valid JSON."""
                response = response.strip()

                # Remove markdown code blocks if present
                if response.startswith("```json"):
                    response = response[7:]
                elif response.startswith("```"):
                    response = response[3:]

                if response.endswith("```"):
                    response = response[:-3]

                # Try to find JSON object bounds
                start_idx = response.find("{")
                end_idx = response.rfind("}")

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response = response[start_idx : end_idx + 1]

                return response.strip()

        return StructuredOutputRunnable(self, parser, schema_dict, include_raw)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        # Construct URL and payload
        url = f"https://api-maestria.onrender.com/maestria_llms/api/v1.0/llm-model/{self.model}"
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "system_message": self.system_message,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(url, json=payload, headers=headers)

            # Check if request was successful
            if resp.status_code == 200:
                response_data = resp.json()

                # For regular text generation, return the response content
                response_text = response_data.get("response", "")
                
                # Ensure we return a string
                if not isinstance(response_text, str):
                    response_text = str(response_text)

                return response_text
            else:
                return f"Error: {resp.status_code}"

        except Exception as e:
            return f"Request failed: {e}"


