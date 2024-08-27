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
import re
from bs4 import BeautifulSoup
import requests
import cloudscraper

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_bluelight_search(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    results = []

    for item in soup.find_all(
        "li", class_="block-row block-row--separated js-inlineModContainer"
    ):
        title_elem = item.find("h3", class_="contentRow-title")
        title = title_elem.text.strip()
        link = title_elem.find("a")["href"]

        author = item.find("a", class_="username").text.strip()

        date = item.find("time")["title"]
        date = re.sub(r" at .*", "", date)  # Remove time from date

        forum = item.find_all("li")[-1].text.strip()

        results.append(
            {
                "title": title,
                "link": link,
                "author": author,
                "date": date,
                "forum": forum,
            }
        )

    return results


def create_markdown_list(results):
    markdown = ""
    for i, result in enumerate(results, 1):
        markdown += f"{i}. [{result['title']}](https://www.bluelight.org{result['link']}) - {result['author']}, {result['date']}\n"
        markdown += f"   Forum: {result['forum']}\n\n"
    return markdown


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
    V2_URL = os.getenv("V2_URL")


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
                logger.debug(f"Context: {context[:500]}")
            else:
                pass
                # query_embedding = await self._get_embedding(query)
                # similar_documents = await self._fetch_similar_documents(query_embedding)
                # context = self._format_context(similar_documents)
                # logger.debug(f"Context: {context}")

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
        for entity in json_data['results']["matches"]:
            name = entity['metadata']['name'].replace('|', '\|')
            type_ = entity['metadata']['type'].replace('|', '\|')
            description = entity['metadata']['description'][:100].replace('|', '\|') + "..."
            context += f"| {name} | {type_} | {description} |\n"
        
        context += "\n## Relationships\n\n"
        context += "| Source | Target | Description |\n"
        context += "|--------|--------|-------------|\n"
        
        # Format relationships
        for relationship in json_data['relationships']["matches"]:
            source = relationship['metadata']['source'].replace('|', '\|')
            target = relationship['metadata']['target'].replace('|', '\|')
            description = relationship['metadata']['description'][:100].replace('|', '\|') + "..."
            context += f"| {source} | {target} | {description} |\n"
        
        return context
    
    def _format_context(self, similar_documents):
        context = ""
        for doc in similar_documents.data:
            context += f"Content: {doc['content']}\n\n##########\n{doc['metadata']['file_name']}\n##########\n\n\n\n"
        return context

    async def _fetch_context_v2(self, query):
        payload = json.dumps({"query": query})
        headers = {"Content-Type": "application/json"}

        response = requests.post(AppConfig.V2_URL, headers=headers, data=payload)
        response_data = self._format_context_for_llm(json_data=response.json())
        return response_data

    def _format_bluelium_search_results(self, query, drug=None):
        scraper = cloudscraper.create_scraper()  # returns a CloudScraper instance
        # scraper.headers.update("Cookie", "xf_csrf=47bDmi_gmA6CIifI; xf_session=lUqHLe0sh1dACAVqa3vzLxPfxWQzGPlY; ")
        # scraper.headers.update("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0")
        # scraper.headers.update("Accept", "application/json, text/javascript, */*; q=0.01")
        #scraper = cloudscraper.CloudScraper()  # CloudScraper inherits from requests.Session
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

    async def _generate_response(self, **kwargs):
        model = kwargs.get("model")

        if model == "openai":
            return await self._generate_openai_response(**kwargs)
        elif model == "experimental":
            return self._generate_experimental_response(**kwargs)
        elif model == "fun":
            return await self._generate_openai_response(fun=True, **kwargs)
        else:  # Default to Gemini
            return await self._generate_gemini_response(
                kwargs["query"], kwargs["context"]
            )

    async def _generate_openai_response(self, **kwargs):
        try:
            query = kwargs.get("query")
            if "!bluelight" in query:
                results = self._format_bluelium_search_results(
                    query=query.split("!bluelight ")[1]
                )
                return results
            context = kwargs.get("context")
            print(context)
            is_drug = kwargs.get("is_drug")
            output_format = kwargs.get("output_format")
            fun = kwargs.get("fun", False)
            dic = """
            ---Drug Information---
                    
                    You have been asked to generate a detailed drug information document in JSON format, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge. Add as much detail as possible. If the tables include a source, make it the source url, otherwise come up with a reliable source yourself. Do NOT cite anything from psychonautwiki.org as the source url, under any circumstances. Ensure all information is accurate and sourced from reliable data.
            """
            messages = [
                {
                    "role": "system",
                    "content": (
                        AppConfig.LLM_SYSTEM
                        if not kwargs.get("fun")
                        else AppConfig.LLM_HUMOROUS_PERSONA
                    )
                    + f"""
                    -- Data Tables --
                    {context if not kwargs.get("fun") else kwargs.get("addl_content", "")}
                    --             --
                    
                    {dic if is_drug else ""}
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    -- USER QUESTION --
                    {query}
                    -- END QUESTION --
                    """,
                },
            ]
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
                search = self._format_bluelium_search_results("", drug=query)
                if "1." not in search:
                    drug_info["trip_reports"] = ""
                    return drug_info
                trs = "\n".join(search.split("\n\n")[:3])
                if output_format == "json":
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
