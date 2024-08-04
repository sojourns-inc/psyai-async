import os
import falcon
from falcon import asgi
from supabase_py_async import create_client
from supabase_py_async.lib.client_options import ClientOptions
from openai import AsyncOpenAI
import base64
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    USER_ID = os.getenv("USER_ID")
    LLM_SYSTEM = os.getenv("LLM_SYSTEM")
    LLM_SUFFIX = os.getenv("LLM_SUFFIX")
    LLM_SUFFIX_BETA = os.getenv("LLM_SUFFIX_BETA")
    EXPERIMENTAL_BASE_URL = os.getenv("EXPERIMENTAL_BASE_URL")
    EXPERIMENTAL_API_KEY = os.getenv("EXPERIMENTAL_API_KEY")
    EXPERIMENTAL_MODEL_NAME = os.getenv("EXPERIMENTAL_MODEL_NAME")
    LLM_HUMOROUS_PERSONA = base64.b64decode(os.getenv("LLM_HUMOROUS_PERSONA")).decode("utf-8")


def initialize_genai_model():
    genai.configure(api_key=AppConfig.GOOGLE_API_KEY)
    generation_config = {
        "temperature": 0.3,
        "top_p": 1,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction=AppConfig.LLM_SYSTEM,
        safety_settings=safety_settings,
    )

supabase_client = None


class StartupMiddleware:
    async def process_startup(self, scope, event):
        global supabase_client
        supabase_client = await create_client(
            AppConfig.SUPABASE_URL, AppConfig.SUPABASE_KEY,
            options=ClientOptions(
            postgrest_client_timeout=15, storage_client_timeout=15)
        )


class PromptResource:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=AppConfig.OPENAI_API_KEY)
        self.experimental_client = AsyncOpenAI(
            base_url=AppConfig.EXPERIMENTAL_BASE_URL,
            api_key=AppConfig.EXPERIMENTAL_API_KEY,
        )
        self.gemini_model = initialize_genai_model()

    async def on_post(self, req, resp):
        try:
            payload = await req.media
            query = payload.get("question", "")
            addl_context = payload.get("context", "")
            temperature = payload.get("temperature", 0)
            tokens = payload.get("tokens", 0)
            model = req.params.get("model", "gemini")

            query_embedding = await self._get_embedding(query)
            similar_documents = await self._fetch_similar_documents(query_embedding)
            context = self._format_context(similar_documents)

            response = await self._generate_response(
                query, context, temperature, tokens, model, addl_context
            )

            resp.media = {"assistant": response}
            resp.status = falcon.HTTP_200
        except Exception as e:
            print(f"Error: {str(e)}")
            resp.media = {"error": "An internal server error occurred"}
            resp.status = falcon.HTTP_500

    async def _get_embedding(self, query):
        try:
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002", input=query, encoding_format="float"
            )
            return embedding_response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise

    async def _fetch_similar_documents(self, query_embedding):
        try:
            response = await supabase_client.rpc(
                "match_vectors",
                {
                    "match_count": 5,
                    "p_user_id": AppConfig.USER_ID,
                    "query_embedding": query_embedding,
                },
            ).execute()
            return response
        except Exception as e:
            print(f"Error fetching similar documents: {str(e)}")
            raise

    def _format_context(self, similar_documents):
        context = ""
        for doc in similar_documents.data:
            context += f"Content: {doc['content']}\n\n##########\n{doc['metadata']['file_name']}\n##########\n\n\n\n"
        return context

    async def _generate_response(
        self, query, context, temperature, tokens, model, addl_context
    ):
        if model == "openai":
            return await self._generate_openai_response(query, context, temperature, tokens)
        elif model == "experimental":
            return await self._generate_experimental_response(
                query, context, temperature, tokens
            )
        elif model == "fun":
            return self._generate_openai_response(
                query, context, temperature, tokens, fun=True, add_context=addl_context
            )
        else:
            return await self._generate_gemini_response(query, context)

    async def _generate_openai_response(
        self, query, context, temperature=0.2, tokens=3000, fun=None, add_context=None
    ):
        try:
            actual_context = (
                f"Here is the chat context:\n\n{add_context if fun else context}"
            )
            actual_query = (
                query
                if fun
                else f"Check your context and find out: {query}\n\n{AppConfig.LLM_SUFFIX}"
            )
            response = await self.openai_client.chat.completions.create(
                model=("gpt-4o" if fun else "gpt-4o"),
                temperature=1.1 if fun else temperature,
                frequency_penalty=0.9 if fun else 0.3,
                presence_penalty=1.0 if fun else 0.0,
                max_tokens=3000 if fun else tokens,
                top_p=1 if fun else 1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            AppConfig.LLM_HUMOROUS_PERSONA
                            if fun
                            else AppConfig.LLM_SYSTEM
                        ),
                    },
                    {"role": "user", "content": actual_context},
                    {"role": "user", "content": actual_query},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating OpenAI response: {str(e)}")
            raise

    async def _generate_gemini_response(self, query, context):
        try:
            convo = self.gemini_model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [f"Here is the chat context:\n\n{context}"],
                    }
                ]
            )
            await convo.send_message_async(
                f"Check your context and find out: {query}\n\n{AppConfig.LLM_SUFFIX_BETA}"
            )
            return convo.last.text
        except Exception as e:
            print(f"Error generating Gemini response: {str(e)}")
            raise

    async def _generate_experimental_response(
        self, query, context, temperature=0.2, tokens=3000
    ):
        try:
            completion = await self.experimental_client.chat.completions.create(
                model=AppConfig.EXPERIMENTAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": AppConfig.LLM_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Here is the chat context:\n\n{context}",
                    },
                    {
                        "role": "user",
                        "content": f"Check your context and find out: {query}\n\n{AppConfig.LLM_SUFFIX}",
                    },
                ],
                temperature=temperature,
                max_tokens=tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating experimental response: {str(e)}")
            raise


app = asgi.App(middleware=[StartupMiddleware()])
app.add_route("/prompt", PromptResource())
