import os
import falcon
from falcon import asgi
from supabase_py_async import create_client
from supabase_py_async.lib.client_options import ClientOptions
from openai import AsyncOpenAI
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import json

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


drug_json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "drug_info",
        "schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The primary name of the substance.",
                },
                "search_url": {
                    "type": "string",
                    "description": "URL for more detailed information on the substance.",
                },
                "chemical_class": {
                    "type": "string",
                    "description": "The chemical class of the substance.",
                },
                "psychoactive_class": {
                    "type": "string",
                    "description": "The psychoactive class of the substance.",
                },
                "dosages": {
                    "type": "object",
                    "properties": {
                        "routes_of_administration": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "route": {
                                        "type": "string",
                                        "description": "The route of administration (e.g., oral, smoked, insufflated).",
                                    },
                                    "units": {
                                        "type": "string",
                                        "description": "Units of measurement (e.g., mg, µg, ml).",
                                    },
                                    "dose_ranges": {
                                        "type": "object",
                                        "properties": {
                                            "threshold": {
                                                "type": "string",
                                                "description": "Threshold dose.",
                                            },
                                            "light": {
                                                "type": "string",
                                                "description": "Light dose.",
                                            },
                                            "common": {
                                                "type": "string",
                                                "description": "Common dose.",
                                            },
                                            "strong": {
                                                "type": "string",
                                                "description": "Strong dose.",
                                            },
                                            "heavy": {
                                                "type": "string",
                                                "description": "Heavy dose.",
                                            },
                                        },
                                        "additionalProperties": False,
                                        "description": "Dosage ranges for the route of administration.",
                                    },
                                },
                                "required": ["route", "units"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "description": "Dosages information for different routes of administration.",
                },
                "duration": {
                    "type": "object",
                    "properties": {
                        "total_duration": {
                            "type": "string",
                            "description": "Total duration of effects.",
                        },
                        "onset": {
                            "type": "string",
                            "description": "Onset time of effects.",
                        },
                        "peak": {
                            "type": "string",
                            "description": "Peak time of effects.",
                        },
                        "offset": {
                            "type": "string",
                            "description": "Offset time of effects.",
                        },
                        "after_effects": {
                            "type": "string",
                            "description": "Duration of after-effects.",
                        },
                    },
                    "description": "Duration details of the substance's effects.",
                },
                "addiction_potential": {
                    "type": "string",
                    "description": "Description of the substance's addiction potential.",
                },
                "interactions": {
                    "type": "object",
                    "properties": {
                        "dangerous": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dangerous drug interactions.",
                        },
                        "unsafe": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Unsafe drug interactions.",
                        },
                        "caution": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Interactions that require caution.",
                        },
                    },
                    "description": "Interaction details for the substance.",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes or warnings about the substance.",
                },
                "subjective_effects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of subjective effects commonly associated with the substance.",
                },
                "tolerance": {
                    "type": "object",
                    "properties": {
                        "full_tolerance": {
                            "type": "string",
                            "description": "Time to full tolerance.",
                        },
                        "half_tolerance": {
                            "type": "string",
                            "description": "Time to half tolerance.",
                        },
                        "zero_tolerance": {
                            "type": "string",
                            "description": "Time to zero tolerance.",
                        },
                        "cross_tolerances": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Substances with cross-tolerance.",
                        },
                    },
                    "description": "Tolerance details for the substance.",
                },
                "half_life": {
                    "type": "string",
                    "description": "Half-life of the substance.",
                },
            },
            "required": ["drug_name"],
            "additionalProperties": False,
        },
    },
}


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
    LLM_HUMOROUS_PERSONA = base64.b64decode(os.getenv("LLM_HUMOROUS_PERSONA")).decode(
        "utf-8"
    )


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
            AppConfig.SUPABASE_URL,
            AppConfig.SUPABASE_KEY,
            options=ClientOptions(
                postgrest_client_timeout=15, storage_client_timeout=15
            ),
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

            # Extract additional parameters
            is_drug = payload.get("drug", False)
            output_format = payload.get("format", "html").lower()

            # Get embeddings and context
            query_embedding = await self._get_embedding(query)
            similar_documents = await self._fetch_similar_documents(query_embedding)
            context = self._format_context(similar_documents)
            logger.debug(f"Context: {context}")

            # Generate response using a dictionary for parameters
            response_params = {
                "query": query,
                "context": context,
                "temperature": temperature,
                "tokens": tokens,
                "model": model,
                "addl_context": addl_context,
                "is_drug": is_drug,
                "output_format": output_format,
            }
            response = await self._generate_response(**response_params)

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
                    "match_count": 10,
                    "p_user_id": AppConfig.USER_ID,
                    "query_embedding": query_embedding,
                },
            ).execute()
            logger.debug(f"Similar documents: {response}")
            return response
        except Exception as e:
            print(f"Error fetching similar documents: {str(e)}")
            raise

    def _format_context(self, similar_documents):
        context = ""
        for doc in similar_documents.data:
            context += f"Content: {doc['content']}\n\n##########\n{doc['metadata']['file_name']}\n##########\n\n\n\n"
        return context

    def _format_drug_info_html(self, drug_json=None):
        # Extracting values from JSON
        drug_name = drug_json.get("drug_name", "Unknown")
        search_url = drug_json.get("search_url", "#")
        chemical_class = drug_json.get("chemical_class", "N/A")
        psychoactive_class = drug_json.get("psychoactive_class", "N/A")
        addiction_potential = drug_json.get("addiction_potential", "N/A")
        notes = drug_json.get("notes", "No additional notes available.")
        half_life_info = drug_json.get("half_life", "N/A")

        # Formatting dosage information
        dosage_info = ""
        if (
            "dosages" in drug_json
            and "routes_of_administration" in drug_json["dosages"]
        ):
            for roa in drug_json["dosages"]["routes_of_administration"]:
                dosage_info += f"- <b>{roa['route']}:</b> "
                dose_ranges = roa.get("dose_ranges", {})
                dose_text = []
                for dose_type, dose_value in dose_ranges.items():
                    dose_text.append(f"{dose_type.capitalize()}: {dose_value}")
                dosage_info += ", ".join(dose_text) + f" {roa['units']}\n"
        else:
            dosage_info = "Dosage information not available."

        # Formatting duration information
        duration_info = ""
        if "duration" in drug_json:
            duration_details = drug_json["duration"]
            duration_info += f"- <b>Total duration:</b> {duration_details.get('total_duration', 'N/A')}\n"
            duration_info += f"- <b>Onset:</b> {duration_details.get('onset', 'N/A')}\n"
            duration_info += f"- <b>Peak:</b> {duration_details.get('peak', 'N/A')}\n"
            duration_info += (
                f"- <b>Offset:</b> {duration_details.get('offset', 'N/A')}\n"
            )
            duration_info += f"- <b>After-effects:</b> {duration_details.get('after_effects', 'N/A')}\n"
        else:
            duration_info = "Duration information not available."

        # Formatting interaction information
        interactions_info = ""
        if "interactions" in drug_json:
            interactions = drug_json["interactions"]
            if "dangerous" in interactions:
                interactions_info += (
                    "<b>Dangerous:</b> " + ", ".join(interactions["dangerous"]) + "\n"
                )
            if "unsafe" in interactions:
                interactions_info += (
                    "<b>Unsafe:</b> " + ", ".join(interactions["unsafe"]) + "\n"
                )
            if "caution" in interactions:
                interactions_info += (
                    "<b>Use with caution:</b> "
                    + ", ".join(interactions["caution"])
                    + "\n"
                )
        else:
            interactions_info = "Interaction information not available."

        # Formatting subjective effects
        subjective_effects = ", ".join(
            drug_json.get(
                "subjective_effects", ["No reported subjective effects available."]
            )
        )

        # Formatting tolerance information
        tolerance_info = ""
        if "tolerance" in drug_json:
            tolerance = drug_json["tolerance"]
            tolerance_info += (
                f"- <b>Full tolerance:</b> {tolerance.get('full_tolerance', 'N/A')}\n"
            )
            tolerance_info += (
                f"- <b>Half tolerance:</b> {tolerance.get('half_tolerance', 'N/A')}\n"
            )
            tolerance_info += (
                f"- <b>Zero tolerance:</b> {tolerance.get('zero_tolerance', 'N/A')}\n"
            )
            tolerance_info += (
                f"- <b>Cross-tolerances:</b> "
                + ", ".join(tolerance.get("cross_tolerances", []))
                + "\n"
            )
        else:
            tolerance_info = "Tolerance information not available."

        # Creating the final info card
        info_card = f"""
    <a href="{search_url}"><b>{drug_name}</b></a>

    🔭 <b>Class</b>
    - ✴️ <b>Chemical:</b> ➡️ {chemical_class}
    - ✴️ <b>Psychoactive:</b> ➡️ {psychoactive_class}

    ⚖️ <b>Dosages</b>
    {dosage_info}

    ⏱️ <b>Duration</b>
    {duration_info}

    ⚠️ <b>Addiction Potential</b> ⚠️
    {addiction_potential}

    🚫 <b>Interactions</b> 🚫
    {interactions_info}

    <b>Notes</b>
    {notes}

    🧠 <b>Subjective Effects</b>
    {subjective_effects}

    📈 <b>Tolerance</b>
    {tolerance_info}

    🕒 <b>Half-life</b>
    {half_life_info}
    """
        return info_card.strip()

    # Example usage:
    # json_data = { ... }  # Insert your JSON data here
    # print(_format_drug_info_html(json_data))

    async def _generate_response(self, **kwargs):
        model = kwargs.get("model")

        if model == "openai":
            return await self._generate_openai_response(**kwargs)
        elif model == "experimental":
            return await self._generate_experimental_response(**kwargs)
        elif model == "fun":
            return await self._generate_openai_response(fun=True, **kwargs)
        else:  # Default to Gemini
            return await self._generate_gemini_response(
                kwargs["query"], kwargs["context"]
            )

    async def _generate_openai_response(self, **kwargs):
        try:
            query = kwargs.get("query")
            context = kwargs.get("context")
            is_drug = kwargs.get("is_drug")
            output_format = kwargs.get("output_format")
            fun = kwargs.get("fun", False)
            actual_context = f'Here is the chat context:\n\n{kwargs.get("addl_content", "") if kwargs.get("fun") else context}'
            actual_query = (
                query
                if kwargs.get("fun")
                else f"Check your context and find out: {query}\n\n{AppConfig.LLM_SUFFIX}"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        AppConfig.LLM_HUMOROUS_PERSONA
                        if kwargs.get("fun")
                        else AppConfig.LLM_SYSTEM
                    ),
                },
                {"role": "user", "content": actual_context},
                {"role": "user", "content": actual_query},
            ]

            if is_drug:
                messages.append(
                    {
                        "role": "system",
                        "content": "Generate a detailed drug information document in JSON format, based on the provided context and query. Add as much detail as possible. If the context includes a source, provide it, otherwise come up with a reliable source yourself. Do NOT cite anything from psychonautwiki.org as a source, under any circumstances. Ensure all information is accurate and sourced from reliable data.",
                    }
                )
            temperature = kwargs.get("temperature", 0.3)
            tokens = kwargs.get("tokens", 3000)
            response = await self.openai_client.chat.completions.create(
                model=("gpt-4o" if fun else "gpt-4o-2024-08-06"),
                temperature=1.1 if fun else temperature,
                frequency_penalty=0.9 if fun else 0.3,
                presence_penalty=1.0 if fun else 0.0,
                max_tokens=3000 if fun else tokens,
                top_p=1 if fun else 1,
                messages=messages,
                response_format=(drug_json_schema if is_drug else None),
            )

            content = response.choices[0].message.content

            if is_drug:
                drug_info = json.loads(content)
                if output_format == "json":
                    return drug_info
                else:  # HTML format
                    return self._format_drug_info_html(drug_json=drug_info)
            else:
                return content

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
