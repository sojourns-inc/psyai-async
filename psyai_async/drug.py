from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class DoseRanges(BaseModel):
    threshold: str = Field(..., description="Threshold dose.")
    light: str = Field(..., description="Light dose.")
    common: str = Field(..., description="Common dose.")
    strong: str = Field(..., description="Strong dose.")
    heavy: str = Field(..., description="Heavy dose.")

class RouteOfAdministration(BaseModel):
    route: str = Field(..., description="The route of administration. SINGLE WORD OR ABBREVIATION ONLY (e.g., oral, smoked, IV, insufflated).")
    units: str = Field(..., description="Units of measurement (e.g., mg, µg, ml).")
    dose_ranges: DoseRanges = Field(..., description="Dosage ranges for the route of administration.")

class Dosages(BaseModel):
    routes_of_administration: List[RouteOfAdministration] = Field(
        default_factory=list, description="Dosages information for different routes of administration."
    )

class Duration(BaseModel):
    total_duration: str = Field(..., description="Total duration of effects.")
    onset: str = Field(..., description="Onset time of effects.")
    peak: str = Field(..., description="Peak time of effects.")
    offset: str = Field(..., description="Offset time of effects.")
    after_effects: str = Field(..., description="Duration of after-effects.")

class Interactions(BaseModel):
    dangerous: List[str] = Field(default_factory=list, description="Dangerous drug interactions.")
    unsafe: List[str] = Field(default_factory=list, description="Unsafe drug interactions.")
    caution: List[str] = Field(default_factory=list, description="Interactions that require caution.")

class Tolerance(BaseModel):
    full_tolerance: str = Field(..., description="Time to full tolerance.")
    half_tolerance: str = Field(..., description="Time to half tolerance.")
    zero_tolerance: str = Field(..., description="Time to zero tolerance.")
    cross_tolerances: List[str] = Field(default_factory=list, description="Substances with cross-tolerance.")

class Citation(BaseModel):
    name: str = Field(..., description="The name or title of the citation.")
    reference: str = Field(..., description="The URL or other reference of the citation.")

# Define the CategoryEnum with the provided categories
class CategoryEnum(str, Enum):
    psychedelic = 'psychedelic'
    dissociative = 'dissociative'
    stimulant = 'stimulant'
    research_chemical = 'research-chemical'
    empathogen = 'empathogen'
    habit_forming = 'habit-forming'
    opioid = 'opioid'
    depressant = 'depressant'
    hallucinogen = 'hallucinogen'
    entactogen = 'entactogen'
    deliriant = 'deliriant'
    antidepressant = 'antidepressant'
    sedative = 'sedative'
    nootropic = 'nootropic'
    barbiturate = 'barbiturate'
    benzodiazepine = 'benzodiazepine'
    supplement = 'supplement'

class DrugInfo(BaseModel):
    drug_name: str = Field(
        ..., 
        description="The primary name of the substance as commonly recognized across various sources, including scientific literature, user reports, and drug databases."
    )
    search_url: str = Field(
        ..., 
        description="URL linking to a comprehensive repository of detailed information about the substance, synthesized from diverse sources including research articles, clinical studies, and user experiences. Must NOT be PsychonautWiki.org URL."
    )
    chemical_class: str = Field(
        ..., 
        description="The chemical class of the substance, identified based on structural and functional similarities with other compounds, as aggregated from chemical databases and research studies."
    )
    psychoactive_class: str = Field(
        ..., 
        description="The psychoactive class of the substance, reflecting its effects on the central nervous system, derived from user reports, pharmacological studies, and expert consensus."
    )
    dosages: Dosages = Field(
        default_factory=Dosages, 
        description="Dosages information for different routes of administration, synthesized from a range of user experiences, clinical guidelines, and pharmacological research, reflecting typical and outlier responses."
    )
    duration: Duration = Field(
        default_factory=Duration, 
        description="Duration details of the substance's effects, including typical onset, peak, and offset times, as reported across multiple studies and user experiences, with variability noted where applicable."
    )
    addiction_potential: str = Field(
        ..., 
        description="A description of the substance's addiction potential, synthesized from epidemiological studies, case reports, and user accounts, reflecting a consensus on the risk of dependency."
    )
    interactions: Interactions = Field(
        default_factory=Interactions, 
        description="Interaction details for the substance, including known dangerous and unsafe combinations, as well as those that require caution, derived from scientific literature and pharmacological databases."
    )
    notes: str = Field(
        ..., 
        description="Additional notes or warnings about the substance, synthesizing a wide range of data points from user experiences, clinical observations, and expert guidelines."
    )
    subjective_effects: List[str] = Field(
        default_factory=list, 
        description="List of subjective effects commonly associated with the substance, aggregated from user reports, clinical studies, and psychopharmacological research, representing typical experiences and outliers."
    )
    tolerance: Tolerance = Field(
        default_factory=Tolerance, 
        description="Tolerance details for the substance, including time to full, half, and zero tolerance, as well as cross-tolerance with other substances, derived from clinical studies and user observations."
    )
    half_life: str = Field(
        ..., 
        description="Half-life of the substance, reflecting the average time for the concentration of the substance to decrease by half in the body, as reported in pharmacokinetic studies."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations supporting the information provided, including names and references, aggregated from scientific literature, user reports, and other reputable sources."
    )
    # New categories field
    categories: List[CategoryEnum] = Field(
        default_factory=list,
        description="List of categories the drug belongs to."
    )

    class Config:
        title = "drug_info"
        extra = "forbid"



legacy_drug_json_schema = {
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
