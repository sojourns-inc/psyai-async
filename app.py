import os
import falcon
from falcon import asgi
from openai import AsyncOpenAI, OpenAI
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import json
import requests
import cloudscraper
from psyai_async.drug import DrugInfo, legacy_drug_json_schema
from psyai_async.formatters import parse_bluelight_search, create_markdown_list
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
    BUDGET_MODEL_NAME = os.getenv("BUDGET_MODEL_NAME")
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    LLM_HUMOROUS_PERSONA = base64.b64decode(os.getenv("LLM_HUMOROUS_PERSONA")).decode(
        "utf-8"
    )
    V2_URL = os.getenv("V2_URL")
    DRUG_INFO_PROMPT = """
    ---Drug Information---
            
    You have been asked to generate a detailed drug information object summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
    Add as much detail as possible. If the tables include a source, make it the source url, otherwise come up with a reliable source yourself.
    Ensure all information is accurate and sourced from reliable data.
    The following sources are FORBIDDEN and you may not cite them as source_url under any circumstances:
    
    -- Forbidden Sources --
    1. PsychonautWiki.org, **or any page on the PsychonautWiki website**
    2. Drugabuse.gov **or any other government website**
    """

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


class PromptResource:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=AppConfig.OPENAI_API_KEY)
        self.experimental_client = OpenAI(
            base_url=AppConfig.EXPERIMENTAL_BASE_URL,
            api_key=AppConfig.EXPERIMENTAL_API_KEY,
        )
        self.gemini_model = initialize_genai_model()
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        # Configuration variables
        self.account_id = AppConfig.CLOUDFLARE_ACCOUNT_ID
        self.api_token = AppConfig.CLOUDFLARE_API_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    async def on_post(self, req, resp):
        try:
            payload = await req.media
            query = payload.get("question", "")
            addl_context = payload.get("context", "")
            temperature = payload.get("temperature", 0)
            tokens = payload.get("tokens", 0)
            model = payload.get("model", "gemini")
            version = payload.get("version", "v1")

            # Extract additional parameters
            is_drug = payload.get("drug", False)
            output_format = payload.get("format", "html").lower()

            # Get embeddings and context
            if version == "v2":
                context = await self._fetch_context_v2(query)
                logger.debug(f"Context: {context}")
            else:
                resp.media = {"error": "Invalid API version."}
                resp.status = falcon.HTTP_400
                return
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

    def _format_context_for_llm(self, json_data):
        context = "## Entities\n\n"
        context += "| Name | Type | Description |\n"
        context += "|------|------|-------------|\n"
        
        # Format entities
        for entity in json_data['entities']["matches"]:
            name = entity['metadata']['name'].replace('|', '\|')
            type_ = entity['metadata']['type'].replace('|', '\|')
            print(entity['metadata']['description'])
            description = entity['metadata']['description'].replace('|', '\|') + "..."
            context += f"| {name} | {type_} | {description} |\n"
        
        context += "\n## Relationships\n\n"
        context += "| Source | Target | Description |\n"
        context += "|--------|--------|-------------|\n"
        
        # Format relationships
        for relationship in json_data['relationships']["matches"]:
            source = relationship['metadata']['source'].replace('|', '\|')
            target = relationship['metadata']['target'].replace('|', '\|')
            description = relationship['metadata']['description'].replace('|', '\|') + "..."
            context += f"| {source} | {target} | {description} |\n"
        
        return context
    
    async def _fetch_context_v2(self, query):
        # Generate embedding for the query
        query_embedding = self.model.encode(query).tolist()

        # Prepare the query payload
        payload = {
            "vector": query_embedding,
            "topK": 20,
            "returnMetadata": "all"
        }

        # Define the URLs for the indices
        entities_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/v2/indexes/psy-entity-index/query"
        relationships_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/v2/indexes/psy-rel-index/query"

        # Query the entities index
        entities_response = requests.post(entities_url, headers=self.headers, data=json.dumps(payload))
        if entities_response.status_code == 200:
            entities_matches = entities_response.json()['result']
        else:
            print(f"Error querying entities index: {entities_response.text}")
            entities_matches = []

        # Query the relationships index
        relationships_response = requests.post(relationships_url, headers=self.headers, data=json.dumps(payload))
        if relationships_response.status_code == 200:
            relationships_matches = relationships_response.json()['result']
        else:
            print(f"Error querying relationships index: {relationships_response.text}")
            relationships_matches = []

        # Format the results into the required JSON structure
        response_data = {
            "query": query,
            "entities": entities_matches,
            "relationships": relationships_matches
        }

        return self._format_context_for_llm(response_data)

    def _format_bluelight_search_results(self, query, drug=None):
        scraper = cloudscraper.create_scraper() 
        if drug:
            html_content = scraper.get(
                f"https://www.bluelight.org/community/search/44/?q=substancecode_{drug.lower()}&o=relevance"
            )
            if html_content is None:
                html_content = scraper.get(
                    f"https://www.bluelight.org/community/search/44/?q={query}&c[title_only]=1&o=relevance"
                )
        else:
            html_content = scraper.get(
                f"https://www.bluelight.org/community/search/44/?q={query}&c[title_only]=1&o=relevance"
            )
        if html_content is not None:
            html_content = html_content.text
        parsed_results = parse_bluelight_search(html_content)
        markdown_list = create_markdown_list(parsed_results)
        return markdown_list

    async def _generate_response(self, **kwargs):
        model = kwargs.get("model")

        if model == "openai":
            return await self._generate_openai_response(**kwargs)
        if model == "openai-next":
            return await self._generate_next_openai_response(**kwargs)
        elif model == "experimental":
            return self._generate_experimental_response(**kwargs)
        elif model == "fun":
            return await self._generate_openai_response(fun=True, **kwargs)
        else:  # Default to Gemini
            return await self._generate_gemini_response(
                kwargs["query"], kwargs["context"]
            )

    async def _generate_next_openai_response(self, **kwargs):
        messages = [
            {
                "role": "user",
                "content":  (
                        AppConfig.LLM_SYSTEM
                        if not kwargs.get("fun")
                        else AppConfig.LLM_HUMOROUS_PERSONA
                    )
                + f"""
                ---Data Tables---
                {kwargs.get("context")}
                ---           ---
                """
            },
            {
                "role": "user",
                "content": f"""
                -- USER QUESTION --
                {kwargs.get("query")}
                -- END QUESTION --
                """,
            },
        ]
        response = await self.openai_client.chat.completions.create(
            model="o1-preview",
            messages=messages,
        )
        return response.choices[0].message.content

    async def _generate_openai_response(self, **kwargs):
        try:
            query = kwargs.get("query")
            if "!bluelight" in query:
                results = self._format_bluelight_search_results(
                    query=query.split("!bluelight ")[1]
                )
                return results
            context = kwargs.get("context")
            print(context)
            is_drug = kwargs.get("is_drug")
            output_format = kwargs.get("output_format")
            fun = kwargs.get("fun", False)

            messages = [
                {
                    "role": "system",
                    "content": (
                        AppConfig.LLM_SYSTEM
                        if not kwargs.get("fun")
                        else AppConfig.LLM_HUMOROUS_PERSONA
                    )
                    + f"""
                    ---Data Tables---
                    {context if not kwargs.get("fun") else kwargs.get("addl_content", "")}
                    ---           ---
                    
                    {AppConfig.DRUG_INFO_PROMPT if is_drug else ""}
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    -- USER QUESTION --
                    {'search_quuery: ' + query if is_drug else query}
                    -- END QUESTION --
                    """,
                },
            ]
            temperature = kwargs.get("temperature", 0.3)
            tokens = kwargs.get("tokens", 4000)
            if output_format == "pyd":
                logger.info(str(temperature), str(tokens), messages)
                response = await self.openai_client.beta.chat.completions.parse(
                    model=AppConfig.DEFAULT_MODEL_NAME,
                    temperature=temperature,
                    max_tokens=tokens,
                    messages=messages,
                    response_format=DrugInfo,
                )
                return response.choices[0].message.parsed.model_dump()
            response = await self.openai_client.chat.completions.create(
                model=(AppConfig.BUDGET_MODEL_NAME if fun else AppConfig.DEFAULT_MODEL_NAME),
                temperature=1.1 if fun else temperature,
                frequency_penalty=0.9 if fun else 0.3,
                presence_penalty=1.0 if fun else 0.0,
                max_tokens=3000 if fun else tokens,
                top_p=1 if fun else 1,
                messages=messages,
                response_format=(legacy_drug_json_schema if is_drug else None),
            )

            content = response.choices[0].message.content

            if is_drug:
                drug_info = json.loads(content)
                search = self._format_bluelight_search_results("", drug=query)
                if "1." not in search:
                    drug_info["trip_reports"] = ""
                    return drug_info
                trs = "\n".join(search.split("\n\n")[:3])
                if output_format == "json" or output_format == "pyd":
                    drug_info["trip_reports"] = trs
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

    def _generate_experimental_response(self, **kwargs):
        try:

            completion = self.experimental_client.chat.completions.create(
                model=AppConfig.EXPERIMENTAL_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": AppConfig.LLM_SYSTEM
                        + f"""
                        -- CONTEXT --
                        {kwargs.get("context")}
                        -- END CONTEXT --
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"""
                        -- USER QUESTION --
                        {kwargs.get("query")}
                        -- END QUESTION --
                        """,
                    },
                ],
                temperature=kwargs.get("temperature", 0.8),
                top_p=1,
                max_tokens=kwargs.get("tokens", 3000),
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating experimental response: {str(e)}")
            raise


app = asgi.App()
app.add_route("/q", PromptResource())
