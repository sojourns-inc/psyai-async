import os
import falcon
from falcon import asgi
from supabase import create_client
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    USER_ID = os.environ.get("USER_ID")
    LLM_SYSTEM = os.environ.get("LLM_SYSTEM")
    LLM_SUFFIX = os.environ.get("LLM_SUFFIX")


def initialize_genai_model():
    genai.configure(api_key=AppConfig.GOOGLE_API_KEY)
    generation_config = {
        "temperature": 0,
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


class PromptResource:
    def __init__(self):
        self.supabase_client = create_client(
            AppConfig.SUPABASE_URL, AppConfig.SUPABASE_KEY
        )
        self.openai_client = OpenAI(api_key=AppConfig.OPENAI_API_KEY)
        self.gemini_model = initialize_genai_model()

    async def on_post(self, req, resp):
        try:
            payload = await req.media
            query = payload["question"]
            model = req.params.get(
                "model", "gemini"
            )  # Default to Gemini if not provided

            query_embedding = self._get_embedding(query)
            similar_documents = self._fetch_similar_documents(query_embedding)
            context = self._format_context(similar_documents)

            if model == "openai":
                response = self._generate_openai_response(query, context)
            else:
                response = self._generate_gemini_response(query, context)

            resp.media = {"assistant": response}
            resp.status = falcon.HTTP_200
        except Exception as e:
            # Log the error and return an appropriate error response
            print(f"Error: {str(e)}")
            resp.media = {"error": "An internal server error occurred"}
            resp.status = falcon.HTTP_500

    def _get_embedding(self, query):
        try:
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002", input=query, encoding_format="float"
            )
            return embedding_response.data[0].embedding
        except Exception as e:
            # Log the error and re-raise the exception
            print(f"Error getting embedding: {str(e)}")
            raise

    def _fetch_similar_documents(self, query_embedding):
        try:
            response = self.supabase_client.rpc(
                "match_vectors",
                {
                    "match_count": 5,
                    "p_user_id": AppConfig.USER_ID,
                    "query_embedding": query_embedding,
                },
            )
            return response.execute()
        except Exception as e:
            # Log the error and re-raise the exception
            print(f"Error fetching similar documents: {str(e)}")
            raise

    def _format_context(self, similar_documents):
        context = ""
        for doc in similar_documents.data:
            context += f"Content: {doc['content']}\n\n##########\n{doc['metadata']['file_name']}\n##########\n\n\n\n"
        return context

    def _generate_openai_response(self, query, context):
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature=0.2,
                max_tokens=3000,
                top_p=1,
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
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log the error and re-raise the exception
            print(f"Error generating OpenAI response: {str(e)}")
            raise

    def _generate_gemini_response(self, query, context):
        try:
            convo = self.gemini_model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [f"""Here is the chat context:\n\n{context}"""],
                    },
                ]
            )
            convo.send_message(
                f"Check your context and find out: {query}\n\n{AppConfig.LLM_SUFFIX}"
            )
            return convo.last.text
        except Exception as e:
            # Log the error and re-raise the exception
            print(f"Error generating Gemini response: {str(e)}")
            raise


app = asgi.App()
app.add_route("/prompt", PromptResource())
