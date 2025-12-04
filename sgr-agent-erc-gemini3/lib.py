import time
import os
from typing import List, Type, TypeVar, Optional
from pydantic import BaseModel
import google.generativeai as genai
from erc3 import ERC3, TaskInfo

T = TypeVar('T', bound=BaseModel)

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

def clean_schema(schema, defs=None):
    if defs is None:
        defs = schema.get('$defs') or schema.get('definitions') or {}
    
    if isinstance(schema, dict):
        # Handle $ref
        if '$ref' in schema:
            ref_path = schema['$ref']
            ref_name = ref_path.split('/')[-1]
            definition = defs.get(ref_name)
            if definition:
                # Merge definition with current schema (excluding $ref)
                resolved = clean_schema(definition, defs)
                merged = resolved.copy()
                for k, v in schema.items():
                    if k == '$ref' or k in ['$defs', 'definitions', 'default', 'title', 'const']:
                        continue
                    merged[k] = clean_schema(v, defs)
                
                # Handle const -> enum conversion
                if 'const' in schema:
                    val = schema['const']
                    if val != "" and val is not None:
                        merged['enum'] = [val]
                
                # Filter existing enums
                if 'enum' in merged:
                    merged['enum'] = [x for x in merged['enum'] if x != "" and x is not None]
                    if not merged['enum']:
                        del merged['enum']
                        
                return merged
        
        # Handle allOf (merge)
        if 'allOf' in schema:
            merged = {}
            for sub in schema['allOf']:
                cleaned_sub = clean_schema(sub, defs)
                if isinstance(cleaned_sub, dict):
                    merged.update(cleaned_sub)
            for k, v in schema.items():
                if k == 'allOf' or k in ['$defs', 'definitions', 'default', 'title', 'const']:
                    continue
                merged[k] = clean_schema(v, defs)
            
            # Handle const -> enum conversion
            if 'const' in schema:
                val = schema['const']
                if val != "" and val is not None:
                    merged['enum'] = [val]
            
            # Filter existing enums
            if 'enum' in merged:
                merged['enum'] = [x for x in merged['enum'] if x != "" and x is not None]
                if not merged['enum']:
                    del merged['enum']

            return merged

        # Handle anyOf (Optional)
        if 'anyOf' in schema:
            options = schema['anyOf']
            non_null = [o for o in options if o.get('type') != 'null']
            if len(non_null) == 1:
                cleaned = clean_schema(non_null[0], defs)
                cleaned['nullable'] = True
                # Merge other properties
                for k, v in schema.items():
                    if k == 'anyOf' or k in ['$defs', 'definitions', 'default', 'title', 'const']:
                        continue
                    cleaned[k] = clean_schema(v, defs)
                
                # Handle const -> enum conversion
                if 'const' in schema:
                    val = schema['const']
                    if val != "" and val is not None:
                        cleaned['enum'] = [val]
                
                # Filter existing enums
                if 'enum' in cleaned:
                    cleaned['enum'] = [x for x in cleaned['enum'] if x != "" and x is not None]
                    if not cleaned['enum']:
                        del cleaned['enum']

                return cleaned
        
        # Standard recursive clean
        cleaned = {}
        
        # Handle const -> enum conversion (local)
        if 'const' in schema:
            val = schema['const']
            if val != "" and val is not None:
                cleaned['enum'] = [val]
            
        for k, v in schema.items():
            if k in ['$defs', 'definitions', 'default', 'title', 'const', 'minItems', 'maxItems', 'minLength', 'maxLength', 'exclusiveMaximum', 'exclusiveMinimum']:
                continue
            cleaned[k] = clean_schema(v, defs)
            
        # Filter existing enums
        if 'enum' in cleaned:
            cleaned['enum'] = [x for x in cleaned['enum'] if x != "" and x is not None]
            if not cleaned['enum']:
                del cleaned['enum']
                
        return cleaned
    
    elif isinstance(schema, list):
        return [clean_schema(item, defs) for item in schema]
    
    return schema

class MyLLM:
    api: ERC3
    task: TaskInfo
    model: str
    max_tokens: int
    chat_session: Optional[genai.ChatSession] = None

    def __init__(self, api: ERC3, model: str, task: TaskInfo, max_tokens=40000) -> None:
        self.api = api
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)

    def query(self, messages: List, response_format: Type[T]) -> T:
        # Extract system prompt and user message from messages list
        # Assuming messages is a list of dicts like [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        # Gemini handles system prompt in model initialization or as the first part of the chat
        
        system_prompt = ""
        user_message = ""
        history = []

        for msg in messages:
            if msg['role'] == 'system':
                system_prompt += msg['content'] + "\n"
            elif msg['role'] == 'user':
                user_message = msg['content']
            elif msg['role'] == 'assistant':
                history.append({"role": "model", "parts": [msg['content']]})
            elif msg['role'] == 'tool':
                 history.append({"role": "user", "parts": [f"Tool Output: {msg['content']}"]})

        schema = response_format.model_json_schema()
        schema = clean_schema(schema)

        gemini_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        # Reconstruct chat history if needed, or just send the last message if we are not maintaining a persistent session object here
        # The original code re-created the chat session or sent messages. 
        # Here we are stateless in `query` similar to the OpenAI implementation, but Gemini is stateful.
        # However, the `messages` argument implies we are passing the full history.
        # So we should construct the history for Gemini.
        
        # Note: The original OpenAI implementation passed the full `messages` list every time.
        # To mimic this with Gemini's chat, we can start a chat with the history (excluding the last user message)
        # and then send the last user message.
        
        if not history and user_message:
             # First turn
             chat = gemini_model.start_chat(history=[])
             prompt = user_message
        else:
             # Subsequent turns
             # We need to be careful with history format. 
             # If we are just doing single-turn generation or managing history manually:
             chat = gemini_model.start_chat(history=history)
             prompt = user_message

        started = time.time()
        
        try:
            response = chat.send_message(prompt)
        except Exception as e:
            # Fallback or re-raise. The original code didn't handle retries in `query` but the sample did in the loop.
            # We'll re-raise to let the caller handle or crash.
            raise e

        if response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count
            )
            self.api.log_llm(
                task_id=self.task.task_id,
                model=f"google/{self.model}",
                duration_sec=time.time() - started,
                usage=usage,
            )

        return response_format.model_validate_json(response.text)
