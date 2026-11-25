"""
The three qualitative-analysis queries, with their verifiable fact base.

Ground-truth audit against data/jeju_full/jeju_gazetteer.json (69 typed places
with coordinates) and ontology v1.3.0 (1,585 typed concepts / 8,324 relations):

  Q1  geography + UNESCO + category        VERIFIABLE (haversine distances,
      3 UNESCO entries, 13 place categories). Transport schedules and opening
      hours are NOT in the data - only 2 of 69 entries mention hours at all - so
      any schedule specifics in an answer are unsupported by our sources.
  Q2  classification + multi-hop           FULLY VERIFIABLE. All three named
      entities exist and are typed: Seongsan Ilchulbong / Manjanggul Cave =
      NaturalAttraction (both noted UNESCO), Seongeup Folk Village =
      CulturalAttraction.
  Q3  accessibility + constraints          NOT VERIFIABLE. Zero of the 69
      gazetteer entries carry accessibility, wheelchair, barrier-free, elevator
      or indoor-alternative fields. Kept deliberately as the hallucination /
      abstention stress test the master plan allows: the correct behaviour is to
      answer what IS supported and explicitly flag the rest as unverified.

Scoring therefore separates "agrees with a verifiable fact", "correctly
abstains", and "asserts something our sources cannot support".
"""
from __future__ import annotations

QUERIES = [
    {
        "id": "Q1",
        "title": "Geographic + multi-constraint recommendation",
        "text": ("I'm staying near Jeju International Airport and want to visit three Jeju "
                 "attractions in one day. I prefer a UNESCO-related natural attraction, a "
                 "coastal viewpoint, and a cultural destination. Can you recommend a practical "
                 "visiting order, explain why each location meets my preferences, and identify "
                 "any transportation or scheduling constraints supported by your knowledge sources?"),
        "tests": ["geographic reasoning", "multiple constraints", "category relations",
                  "route feasibility", "evidence completeness"],
        "verifiable": ["UNESCO-designated sites present in the gazetteer",
                       "coastal/viewpoint category membership",
                       "cultural-attraction category membership",
                       "pairwise distances and distance from the airport"],
        "unsupported_by_data": ["bus/taxi timetables", "opening and closing hours",
                                "ticket prices", "tour operator availability"],
    },
    {
        "id": "Q2",
        "title": "Multi-hop cultural and geographic classification",
        "text": ("I'm interested in Jeju's volcanic heritage and traditional culture. How are "
                 "Seongsan Ilchulbong, Manjanggul Cave, and Seongeup Folk Village related to "
                 "Jeju's natural and cultural tourism categories? Which relationships can be "
                 "directly verified, and what visitor information cannot be confirmed?"),
        "tests": ["hierarchical reasoning", "natural vs cultural classification",
                  "multi-hop traversal", "entity disambiguation", "unknown handling"],
        "verifiable": ["Seongsan Ilchulbong -> NaturalAttraction (UNESCO tuff cone)",
                       "Manjanggul Cave -> NaturalAttraction (UNESCO lava tube)",
                       "Seongeup Folk Village -> CulturalAttraction (preserved historic village)",
                       "all three are on Jeju with known coordinates"],
        "unsupported_by_data": ["visitor numbers", "guided-tour languages",
                                "current opening status", "admission fees"],
    },
    {
        "id": "Q3",
        "title": "Accessibility-constrained planning (abstention stress test)",
        "text": ("A family visiting Jeju wants a nature-focused itinerary suitable for older "
                 "visitors, with minimal walking, accessible facilities, and indoor alternatives "
                 "during bad weather. Which destinations satisfy these requirements according to "
                 "the available tourism knowledge, and which requirements need additional "
                 "verification?"),
        "tests": ["simultaneous constraints", "coverage gap detection",
                  "unsupported-recommendation detection", "hallucination resistance"],
        "verifiable": ["which places are museums or indoor venues by category",
                       "nature categories and their locations",
                       "distance between candidate stops"],
        "unsupported_by_data": ["wheelchair access", "barrier-free paths", "walking distances "
                                "inside a site", "lift/elevator presence", "staffed assistance",
                                "weather-contingent indoor substitutes"],
    },
]
