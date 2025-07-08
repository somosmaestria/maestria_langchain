# Maestria LangChain

A custom LangChain LLM wrapper for Maestria LLMs API, providing seamless integration with LangChain's ecosystem and built-in support for structured output generation.

## Features

- **LangChain Integration**: Full compatibility with LangChain's LLM interface
- **Structured Output**: Built-in support for JSON and Pydantic model outputs
- **Flexible Configuration**: Customizable model parameters and system messages
- **Error Handling**: Robust response parsing with multiple fallback strategies
- **Type Safety**: Full type hints and Pydantic validation

## Installation

```bash
pip install -r requirements.txt
```

Or install directly from the repository:

```bash
pip install git+https://github.com/somosmaestria/maestria_langchain.git@main
```

## Quick Start

### Basic Usage

```python
from maestria_langchain import MaestriaLLM

# Initialize the LLM
llm = MaestriaLLM(
    model="deepseek-r1:8b",
    max_tokens=500,
    temperature=0.7,
    api_key="your-api-key"  # Optional
)

# Generate text
response = llm("What is the capital of France?")
print(response)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field
from maestria_langchain import MaestriaLLM

# Define your output schema
class PersonInfo(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's job or profession")
    hobbies: list[str] = Field(description="List of hobbies")

# Create structured output chain
llm = MaestriaLLM(model="deepseek-r1:8b", temperature=0.3)
structured_llm = llm.with_structured_output(PersonInfo)

# Generate structured response
prompt = "Tell me about a fictional character: a 30-year-old software engineer named Alice"
result = structured_llm.invoke(prompt)

print(f"Name: {result.name}")
print(f"Age: {result.age}")
print(f"Occupation: {result.occupation}")
print(f"Hobbies: {', '.join(result.hobbies)}")
```

### Structured Output with Dictionary Schema

```python
from maestria_langchain import MaestriaLLM

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "confidence", "keywords"]
}

llm = MaestriaLLM()
structured_llm = llm.with_structured_output(schema)

result = structured_llm.invoke("I love this new product! It's amazing and works perfectly.")
print(result)
# Output: {'sentiment': 'positive', 'confidence': 0.95, 'keywords': ['love', 'amazing', 'perfectly']}
```

### Using with LangChain Chains

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from maestria_langchain import MaestriaLLM

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief summary about {topic} in exactly 3 sentences."
)

# Create LLM and chain
llm = MaestriaLLM(model="deepseek-r1:8b", max_tokens=200)
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(topic="artificial intelligence")
print(result)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "deepseek-r1:8b" | The model name to use |
| `api_key` | Optional[str] | None | API key for authentication |
| `max_tokens` | int | 100 | Maximum number of tokens to generate |
| `temperature` | float | 0.1 | Controls randomness (0.0 to 1.0) |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `stream` | bool | False | Whether to stream responses |
| `system_message` | Optional[str] | None | System message for the conversation |

## Structured Output Methods

### `with_structured_output()`

Transform the LLM to return structured data matching a specified schema.

```python
def with_structured_output(
    self,
    schema: Union[Dict, Type[BaseModel]],
    *,
    method: Literal["json_mode"] = "json_mode",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]
```

**Parameters:**
- `schema`: Either a Pydantic BaseModel class or a dictionary schema
- `method`: Output formatting method (currently only "json_mode" supported)
- `include_raw`: If True, returns both raw and parsed output
- `**kwargs`: Additional keyword arguments

**Returns:**
A Runnable that takes string inputs and returns structured output.

### Error Handling with `include_raw=True`

```python
structured_llm = llm.with_structured_output(PersonInfo, include_raw=True)
result = structured_llm.invoke("Tell me about John, a teacher")

if result["parsing_error"] is None:
    person = result["parsed"]
    print(f"Successfully parsed: {person.name}")
else:
    print(f"Parsing failed: {result['parsing_error']}")
    print(f"Raw response: {result['raw']}")
```

## API Integration

The library integrates with the Maestria LLMs API at `https://api-maestria.onrender.com/maestria_llms/api/v1.0/llm-model/{model}`.

### Authentication

If an API key is provided, it will be sent as a Bearer token in the Authorization header:

```python
llm = MaestriaLLM(
    model="deepseek-r1:8b",
    api_key="your-api-key-here"
)
```

### Error Handling

The library handles various error conditions:

- **HTTP errors**: Returns descriptive error messages with status codes
- **Network errors**: Returns "Request failed" with exception details
- **JSON parsing errors**: Multiple fallback strategies for malformed responses
- **Schema validation errors**: Clear error messages for structured output failures

## Advanced Examples

### Custom System Message

```python
llm = MaestriaLLM(
    model="deepseek-r1:8b",
    system_message="You are a helpful assistant specialized in Python programming.",
    temperature=0.2
)

response = llm("How do I implement a binary search in Python?")
```

### Chaining with Multiple Structured Outputs

```python
from pydantic import BaseModel

class CodeAnalysis(BaseModel):
    complexity: str = Field(description="Time complexity (e.g., O(n))")
    space_complexity: str = Field(description="Space complexity")
    strengths: list[str] = Field(description="Code strengths")
    improvements: list[str] = Field(description="Suggested improvements")

class CodeSummary(BaseModel):
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    difficulty: str = Field(description="Beginner, Intermediate, or Advanced")

llm = MaestriaLLM(model="deepseek-r1:8b", temperature=0.3)

# First chain: analyze code
analyzer = llm.with_structured_output(CodeAnalysis)
code = "def binary_search(arr, x): ..."
analysis = analyzer.invoke(f"Analyze this code: {code}")

# Second chain: summarize code
summarizer = llm.with_structured_output(CodeSummary)
summary = summarizer.invoke(f"Summarize this code: {code}")

print(f"Complexity: {analysis.complexity}")
print(f"Purpose: {summary.purpose}")
```

## Requirements

- Python ≥ 3.9
- langchain
- pydantic
- requests

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.

## Author

**Jonathan Hidalgo** - [jonathan@ilustraviz.com](mailto:jonathan@ilustraviz.com)

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/somosmaestria/maestria_langchain).