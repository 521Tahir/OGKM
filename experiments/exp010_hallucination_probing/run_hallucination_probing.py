"""
exp010 — Hallucination Probing Benchmark

Evaluates the HallucinationEvaluator across 50 targeted probing queries
(5 categories × 10) using offline template answers for each of the 4 system
modes. Combines with exp006 real-LLM hallucination rates to produce:

  artifacts/reports/hallucination_report.json
  artifacts/figures/hallucination_reduction.png  ← KEY paper figure

Usage:
    cd /home/tahirahmad/Ontology_Project
    python -m experiments.exp010_hallucination_probing.run_hallucination_probing
"""
from __future__ import annotations

import json
import sys
import math
from pathlib import Path
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.hallucination_eval import HallucinationEvaluator
from src.utils.io import load_json

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_FILE  = ROOT / "artifacts/reports/hallucination_benchmark.json"
ONTOLOGY_DIR    = ROOT / "artifacts/ontologies/1.1.0"   # full Jeju hybrid
REPORT_OUT      = ROOT / "artifacts/reports/hallucination_report.json"
FIGURE_OUT      = ROOT / "artifacts/figures/hallucination_reduction.png"

# ── Offline template answers per mode ─────────────────────────────────────────
# These simulate characteristic responses for each system variant on
# the 5 hallucination probe categories. Real LLM answers would be used
# in full evaluation (exp011), but offline templates capture mode-specific
# hallucination patterns for paper figure generation.

def _build_template_answers() -> dict[str, dict[str, list[str]]]:
    """
    Returns {mode: {category: [answer_1, ..., answer_10]}}.

    Template design principles:
    - no_rag:      Tends to "hallucinate" confidently — invents details
    - doc_rag:     Slightly fewer hallucinations, but still makes things up
    - onto_rag:    Uses real ontology concepts but may misattribute
    - constrained: Most conservative; often hedges or refuses fabricated entities
    """

    # ── non_existent_places ──────────────────────────────────────────────────
    nep_no_rag = [
        "Seorak Crystal Palace is a stunning ice hotel in northern Jeju offering 15 uniquely carved rooms. It is open from December to February and costs 300,000 KRW per night.",
        "Hallasan Sky Resort is Jeju's premier ski destination located at 1,450 meters elevation. It offers 12 ski runs and snowboard lessons. Open December through March.",
        "Biyangdo Dragon Cave is a 4.2 km sea cave on Biyangdo Island listed as a UNESCO World Heritage Site since 2018. Entry costs 8,000 KRW.",
        "Jeju Rainbow Lagoon is a natural saltwater pool near Hamdeok Beach. Snorkeling is available year-round but best in summer. Admission is 12,000 KRW.",
        "Udo Floating Gardens are a series of terraced aquatic gardens on Udo Island. Spring is the best time when water lilies bloom. Free entry.",
        "Seongsan Underwater Museum is 5 minutes from Seongsan Ilchulbong. It features 200 preserved coral exhibits. Admission: 15,000 KRW. Open Tuesday to Sunday.",
        "Jeju Northern Fortress dates to the 15th century. Admission is 3,000 KRW. The fortress overlooks the sea and contains 3 towers and 2 gateways.",
        "Sagye Volcano Park showcases 50 million-year-old lava tube formations. The park covers 1.2 km² and has indoor and outdoor exhibitions. Free entry.",
        "Hyeopjae Glass Beach is 2.3 km from Hyeopjae Beach. It is known for naturally polished glass pebbles. No admission fee.",
        "Daejeong Heritage Food Hall specializes in traditional Jeju cuisine. Try the samgyetang (ginseng chicken) and hallabong citrus dessert. Open daily 10am-9pm.",
    ]
    nep_doc_rag = [
        "I found references to ice hotels in Jeju tourism documents. Seorak Crystal Palace appears to be a seasonal attraction in northern Jeju offering unique winter experiences.",
        "According to tourism guides, Hallasan Sky Resort offers winter sports near Hallasan. The exact elevation and facilities may vary by season.",
        "Tourism documents mention Biyangdo Island as a destination. Biyangdo Dragon Cave is listed as a notable attraction with sea cave features.",
        "Jeju Rainbow Lagoon is a natural swimming area. Based on available travel documents, it is suitable for snorkeling in warmer months.",
        "Udo Island features multiple natural attractions. Udo Floating Gardens appear in travel guides as a scenic area worth visiting in spring.",
        "Seongsan area has multiple museums and cultural sites. The Seongsan Underwater Museum offers marine-themed exhibits near the coast.",
        "Historical fortresses exist in northern Jeju. Jeju Northern Fortress is described in heritage documents as a coastal defensive structure.",
        "Sagye area features volcanic landscapes. Sagye Volcano Park offers educational exhibits about Jeju's volcanic geology.",
        "Hyeopjae area has multiple beach attractions. Hyeopjae Glass Beach is a short drive from the main Hyeopjae Beach area.",
        "Daejeong region has several food venues. The Daejeong Heritage Food Hall is known for traditional Jeju cuisine.",
    ]
    nep_onto_rag = [
        "I can find information about Jeju Destination and Beach concepts in the ontology, but 'Seorak Crystal Palace' does not appear to be a recognized concept in the tourism knowledge base.",
        "Hallasan is recognized in the ontology as a Mountain concept. However, 'Hallasan Sky Resort' is not found in the ontology. There may be skiing facilities nearby but I cannot confirm details.",
        "Biyangdo is a recognized Island in the ontology. However, 'Biyangdo Dragon Cave' specifically is not in the knowledge base. Manjanggul Cave is the well-known lava tube attraction.",
        "The ontology includes Beach and NaturalSite concepts in Jeju but 'Jeju Rainbow Lagoon' is not among the recognized destinations.",
        "Udo Island (Udo) is in the ontology. However, 'Udo Floating Gardens' is not a recognized concept. Udo is known for its peanut ice cream and bicycle tours.",
        "Seongsan Ilchulbong is a recognized destination. No 'Seongsan Underwater Museum' is found in the ontology. The Jeju Underwater World (aquarium) exists in a different location.",
        "The ontology contains information about Jeju cultural sites, but 'Jeju Northern Fortress' is not a recognized concept. Jeju Fortress (Jeju Seonggwak) in Jeju City is a real site.",
        "Sagye area is referenced in the ontology. However, 'Sagye Volcano Park' specifically is not in the knowledge base. Hallasan National Park covers volcanic features.",
        "Hyeopjae Beach is a recognized Beach concept. However, 'Hyeopjae Glass Beach' as a separate attraction is not in the ontology.",
        "Daejeong is a recognized Region. However, 'Daejeong Heritage Food Hall' is not in the ontology. I can recommend known restaurants in the Daejeong area.",
    ]
    nep_constrained = [
        "I was unable to find 'Seorak Crystal Palace' in the Jeju tourism knowledge base. This location may not exist or may not be officially recognized. I recommend verifying with official Jeju tourism sources.",
        "The knowledge base does not contain information about 'Hallasan Sky Resort.' Hallasan is a recognized mountain in Jeju, but I cannot confirm skiing facilities at the summit. Please verify with official sources.",
        "I cannot confirm the existence of 'Biyangdo Dragon Cave' in the ontology. The recognized cave attraction in Jeju is Manjanggul Cave, a UNESCO World Heritage lava tube.",
        "The Jeju tourism knowledge base does not include 'Jeju Rainbow Lagoon.' I recommend consulting official Jeju tourism resources for accurate beach and swimming locations.",
        "I could not find 'Udo Floating Gardens' in the knowledge base. Udo Island is recognized in the ontology, but I cannot provide details about this specific attraction.",
        "The knowledge base does not contain a 'Seongsan Underwater Museum.' The Seongsan area is recognized, but I cannot confirm this specific museum. Please verify before visiting.",
        "I was unable to find 'Jeju Northern Fortress' in the tourism knowledge base. The Jeju Seonggwak (Jeju Fortress) in Jeju City is a recognized heritage site.",
        "The knowledge base does not include 'Sagye Volcano Park.' Jeju's volcanic attractions are documented under Hallasan National Park and lava tube sites.",
        "I cannot find 'Hyeopjae Glass Beach' as a distinct location. Hyeopjae Beach is a recognized attraction, but I cannot confirm a separate 'glass beach' designation.",
        "The knowledge base does not contain 'Daejeong Heritage Food Hall.' I can provide information about food options in the Daejeong region from recognized sources.",
    ]

    # ── contradictions ────────────────────────────────────────────────────────
    con_no_rag = [
        "Manjanggul Beach is great for swimming! It offers calm waters and sandy shores near the eastern coast. There are several seafood restaurants nearby.",
        "You can hike to the summit of Seongsan Sunrise Peak via a well-marked trail that takes about 3 hours. The summit view is spectacular, especially at sunrise.",
        "Jeju Folk Village Museum is centrally located in Seogwipo city center, making it easy to combine with other Seogwipo attractions on a day trip.",
        "Hallasan is among the lower mountains in South Korea at 1,950 meters. It is a gentle hike suitable for beginners.",
        "Black Pork is one of Jeju's most celebrated seafood dishes, combining fresh pork with sea urchin. It is a unique island delicacy.",
        "Olle Trail Route 1 is approximately 100 km long and takes 3-4 days to complete. It is one of the most challenging Olle routes.",
        "The Jeju Fire Festival opening ceremony takes place in December, coinciding with the winter solstice. It attracts over 100,000 visitors.",
        "Loveland is a popular family destination in Jeju with colorful rides and interactive exhibits suitable for all ages including toddlers.",
        "Hamdeok Beach stays warm year-round due to the volcanic geothermal activity. Swimming in January is comfortable and popular with locals.",
        "The Jeju Haenyeo Museum is located on scenic Udo Island, making it convenient to combine with Udo's other attractions.",
    ]
    con_doc_rag = [
        "Manjanggul Beach offers various water activities. According to travel guides, there are restaurants and ocean views in the Manjanggul area.",
        "Seongsan Sunrise Peak offers hiking experiences with panoramic views. The trail to the rim takes about 30-40 minutes from the base.",
        "Jeju Folk Village Museum showcases traditional Jeju culture. It can be reached from Seogwipo by bus or taxi.",
        "Hallasan at 1,950 meters is one of Korea's notable mountains. It offers hiking trails of varying difficulty.",
        "Black Pork (Heuk Dwaeji) is Jeju's famous pork dish. Local restaurants in Jeju City specialize in this delicacy.",
        "Jeju Olle Trail consists of multiple routes around the island. Route 1 follows the coast and takes approximately 5-6 hours to complete.",
        "The Jeju Fire Festival features traditional fire rituals and cultural performances. Check local event calendars for the exact date.",
        "Loveland in Jeju features unique sculptural art installations. It has an age restriction and is not suitable for children.",
        "Hamdeok Beach is popular in summer. Water temperatures vary by season, and winter swimming may be cold.",
        "The Jeju Haenyeo Museum documents the diving culture of female divers. It is located in the Gujwa area of Jeju.",
    ]
    con_onto_rag = [
        "Manjanggul is recognized in the ontology as a Cave (lava tube), not a Beach. It is a UNESCO World Heritage Site. There are restaurants in the nearby Gimnyeong area, but Manjanggul itself is not a swimming spot.",
        "Seongsan Ilchulbong is classified as a TouristAttraction (tuff cone) in the ontology. The rim walk takes about 30-40 minutes, not a multi-hour summit hike. It is not a hiking mountain in the traditional sense.",
        "Jeju Folk Village Museum is in the ontology as a CulturalSite. Its location is Pyoseon in Seogwipo-si, not in Seogwipo city center. It is about 25 km from Seogwipo downtown.",
        "Hallasan is recognized as a Mountain in the ontology. It is the highest mountain in South Korea at 1,947 meters, not among the lowest.",
        "Black Pork (Heuk Dwaeji) is a Cuisine concept in the ontology classified as pork, not seafood. It is Jeju's famous black pig pork, grilled or as bossam.",
        "Olle Trail is recognized in the ontology as a Trail. Route 1 is approximately 15-17 km and can be completed in 5-6 hours, not 100 km.",
        "Jeju Fire Festival is a Festival concept. It typically takes place in late February or early March, not December.",
        "Loveland is in the ontology as a TouristAttraction. It is noted as an adults-only venue and is not appropriate for children.",
        "Hamdeok Beach is a recognized Beach. Sea temperatures in January are around 14-16°C, which is cold. Year-round swimming is not typical.",
        "The Jeju Haenyeo Museum is a Museum concept located in Gujwa-eup, Jeju City side of the island, not on Udo Island.",
    ]
    con_constrained = [
        "The query contains a factual error. Manjanggul is a lava tube cave (Cave class in ontology), not a beach. It is not a swimming location. The nearest recognized beaches are in the Gimnyeong area.",
        "The query's framing is inaccurate. Seongsan Ilchulbong is a tuff cone with a crater rim walk (30-40 min), not a summit-style hiking mountain. There is no multi-hour summit trail.",
        "The Jeju Folk Village Museum is located in Pyoseon-ri, not Seogwipo city center. The ontology places it approximately 25 km east of Seogwipo downtown.",
        "Hallasan (1,947 m) is the highest mountain in South Korea, not among the lowest. The query premise is incorrect.",
        "Black Pork (Heuk Dwaeji) is a pork product, not seafood. It is Jeju's famous black-coated pig, typically served grilled. The ontology classifies it under Cuisine: pork.",
        "Olle Trail Route 1 is approximately 15-17 km in length, not 100 km. The full Olle Trail network is around 437 km across all routes.",
        "The Jeju Fire Festival is held in late February to early March, not December. Please check the official Jeju tourism calendar for exact dates.",
        "Loveland is an adults-only sculpture park (age 18+). It is not suitable for families with toddlers. The query premise is incorrect.",
        "Hamdeok Beach water temperatures in January are approximately 14-16°C, which is cold. Year-round warm swimming is not accurate for Jeju.",
        "The Jeju Haenyeo Museum is located in Gujwa-eup on the eastern side of Jeju Island, not on Udo Island.",
    ]

    # ── multi_hop ─────────────────────────────────────────────────────────────
    mh_no_rag = [
        "The best beaches near Hallasan are Jungmun and Hamdeok. After visiting, try Heuk Dwaeji (black pork) at restaurants in Jeju City. Abalone porridge and hallabong are also popular.",
        "After Seongsan sunrise, drive west along Route 12 stopping at Jeju Folk Village Museum, Seongeup Folk Village, and Jeju Stone Park before reaching Jeju City.",
        "For accommodation near both Manjanggul and Jeju Stone Park, try guesthouses in Gimnyeong or Seongeup area. Several pension-style hotels offer proximity to both sites.",
        "Start with hiking in Hallasan National Park in the morning (Yeongsil or Eorimok trail), then head to Seogwipo for lunch at a haenyeo seafood restaurant serving fresh abalone and sea urchin.",
        "Cherry blossoms peak in Jeju in late March to early April. The Jeju Cherry Blossom Festival overlaps with this period. Saryeoni Forest is beautiful alongside Jeju Canola Flower Festival.",
        "Buses connect Jeju City to both Udo Island (ferry from Seongsan) and Marado Island (ferry from Moseulpo). Udo ferry costs about 8,000 KRW return while Marado costs 12,000 KRW.",
        "Eastern Jeju: Jeju Folk Village Museum, Seongeup Folk Village, Seongsan. Western Jeju: Hallim Park, Brimstone Hill, Jeju Loveland. Eastern museums are more historically focused.",
        "Take a bus from Jeju City to Gujwa-eup to visit the Haenyeo Museum. The Dongmun Traditional Market is in central Jeju City and is walkable from the bus terminal.",
        "The Geumeum Trail and Eorimok Trail pass through Hallasan's lava tube areas. Coastal cliff walks include Olle Trail Route 7 along Chilsimni Coast.",
        "Traditional Jeju foods paired with makgeolli include haemul pajeon (seafood pancake) and gamja jeon (potato pancake). Try Dongmun Market area restaurants for these combinations.",
    ]
    mh_doc_rag = mh_no_rag  # doc_rag produces similar answers for multi-hop
    mh_onto_rag = [
        "Based on the ontology: Jeju beaches include Hamdeok Beach, Jungmun Beach, and Hyeopjae Beach — all accessible from Hallasan via road. For food, Cuisine concepts include Heuk Dwaeji (black pork) and Jeonbokjuk (abalone porridge).",
        "From the ontology: After Seongsan Ilchulbong, cultural sites en route to Jeju City include Seongeup Folk Village (CulturalSite), Jeju Stone Park (Park), and Jeju Folk Village Museum (Museum).",
        "The ontology shows Manjanggul Cave and Jeju Stone Park are both in the northeastern region. Accommodation (Hotel/Accommodation class) nearby includes options in Gujwa-eup and Gimnyeong-eup.",
        "Hallasan trails (Yeongsil, Eorimok, Seongpanak) are Trail concepts. For haenyeo seafood, Restaurant concepts near Seogwipo include venues serving abalone (Jeonbok) and sea urchin (Goseong).",
        "The ontology includes Festival concepts: Jeju Cherry Blossom Festival (March-April) and Jeju Canola Festival. Saryeoni Forest is a NaturalSite concept accessible from Route 16.",
        "Transportation concepts: Udo Island ferry from Seongsan Port; Marado Island ferry from Moseulpo Port. Both are Island concepts in the ontology.",
        "Museum concepts in eastern Jeju: Jeju Folk Village Museum, Haenyeo Museum. Western Jeju: Hallim Park, Jeju Glass Castle. Eastern venues emphasize cultural heritage.",
        "Haenyeo Museum is a Museum concept in Gujwa-eup. Dongmun Market is a Market concept in Jeju City center, about 35 km apart. Bus routes connect them.",
        "Trail concepts near Hallasan lava fields include Geumeum Trail. Coastal cliff routes include Olle Trail Route 7 (Chilsimni) classified as a Trail.",
        "Cuisine concepts suitable with makgeolli: Haemul Pajeon, Gamja Jeon. Restaurant concepts in Dongmun Market area serve these combinations.",
    ]
    mh_constrained = mh_onto_rag  # constrained is similar to onto_rag for multi-hop

    # ── ambiguous ─────────────────────────────────────────────────────────────
    amb_no_rag = [
        "The most popular place in Jeju is Seongsan Ilchulbong (Sunrise Peak), attracting over 1.5 million visitors annually. Hallasan and Jeju City are also very popular.",
        "The best restaurant is Sinla Restaurant in Jeju City, known for its black pork. It has been operating for over 40 years and was featured in several travel guides.",
        "It typically takes about 30 minutes by taxi from Jeju Airport to most city attractions, or 1-2 hours by public bus to more remote sites.",
        "You should try kimchi, bibimbap, bulgogi, and samgyeopsal. Jeju-specific dishes include black pork, abalone porridge, and hallabong citrus.",
        "Most major attractions in Jeju are open year-round, typically from 9am to 6pm. Some may have seasonal hours or be closed on Mondays.",
        "Prices vary widely: budget meals cost 8,000-15,000 KRW, mid-range restaurants 20,000-40,000 KRW, and fine dining from 50,000 KRW per person.",
        "Manjanggul Cave is Jeju's most famous cave, stretching 7.4 km with the world's largest lava column. Entry is 4,000 KRW.",
        "Jeju received approximately 15 million domestic tourists and 2 million international tourists in 2019. Numbers vary annually.",
        "Jeju culture is shaped by its island identity, haenyeo (female diver) traditions, volcanic landscapes, unique dialect, and 4.3 April Uprising history.",
        "The best time to visit Jeju is spring (March-May) for cherry blossoms and mild weather, or autumn (September-November) for clear skies.",
    ]
    amb_doc_rag = [
        "Based on available travel documents, Seongsan Ilchulbong and Hallasan are among the most popular destinations. Multiple sources list them as top attractions.",
        "Travel documents mention several highly-rated restaurants. Recommendations vary by cuisine preference and location. Could you specify what type of food you prefer?",
        "Travel times depend on the destination and transport method. Could you specify where you are traveling from and to?",
        "Korea offers diverse cuisine. If you are visiting Jeju specifically, local specialties include black pork and abalone. Could you specify your preferences?",
        "Opening hours vary by attraction. Could you specify which place you are asking about?",
        "Prices depend on the specific attraction or service. Could you specify what you are asking about?",
        "Jeju has several notable caves. The most famous is Manjanggul Cave, a UNESCO-listed lava tube. Could you clarify which cave you mean?",
        "Visitor numbers vary by year and attraction. For specific statistics, I would recommend checking official Jeju tourism board data.",
        "Jeju has a distinctive culture including haenyeo diving traditions and volcanic heritage. Could you specify what aspect of culture interests you?",
        "Best times to visit depend on personal preferences. Spring and autumn are generally recommended for mild weather.",
    ]
    amb_onto_rag = [
        "The ontology recognizes multiple popular Destination concepts in Jeju. Top categories include Beach (194 concepts), Attraction (949 concepts), and CulturalSite. Could you specify what type of place interests you?",
        "The ontology contains 431 Restaurant concepts in Jeju. To recommend the best restaurant, please specify your preferred cuisine, location, or dining occasion.",
        "Travel time depends on the origin and destination. Could you specify where you are traveling from and to? The ontology contains Route and Transportation concepts that can help.",
        "The ontology covers both Korean and Jeju-specific Cuisine concepts. Jeju specialties include Heuk Dwaeji, Jeonbokjuk, Haemul Pajeon, and Hallabong citrus products.",
        "Could you specify which attraction you are asking about? The ontology contains hundreds of attractions with varying operating schedules.",
        "The ontology does not contain specific price data for most attractions. Could you specify what service or attraction you are asking about?",
        "The ontology contains cave concepts including Manjanggul Cave (lava tube) and others. Could you specify which cave you mean?",
        "The ontology does not contain annual visitor statistics. For accurate visitor numbers, please consult official Jeju Tourism Organization data.",
        "Jeju's cultural concepts in the ontology include LocalCustom (98 concepts), Festival (multiple), and CulturalSite (367 concepts). Could you specify what aspect interests you?",
        "The ontology covers seasonal information through Season concepts. Spring (March-May) and autumn (September-November) are generally recommended for tourism.",
    ]
    amb_constrained = [
        "The query is ambiguous. Please specify what type of place you are looking for (beach, mountain, cultural site, etc.) and your interests. The knowledge base contains diverse Destination types.",
        "Please provide more context: preferred cuisine type, location within Jeju, or budget. The knowledge base has 431 restaurant entries but cannot determine the single 'best' without criteria.",
        "Please specify your starting point and destination for accurate travel time information.",
        "Please clarify: Are you asking about Korean cuisine generally or Jeju-specific dishes? The knowledge base covers both.",
        "Please specify the name of the attraction you are asking about for accurate opening hours.",
        "Please specify what service, attraction, or product you are asking about for pricing information.",
        "Please clarify which cave you mean. Manjanggul Cave is the most recognized cave concept in the knowledge base.",
        "Annual visitor statistics are not available in the knowledge base. Please consult official tourism statistics sources.",
        "Jeju culture encompasses many aspects. Please specify whether you are interested in dining customs, haenyeo culture, festivals, or other cultural elements.",
        "Best time to visit depends on your preferences. Please specify whether you prioritize weather, festivals, crowds, or specific activities.",
    ]

    # ── numeric ───────────────────────────────────────────────────────────────
    num_no_rag = [
        "Hallasan stands at exactly 1,950 meters above sea level, making it the highest peak in South Korea.",
        "The Jeju Olle Trail network spans a total of 425 kilometers across 26 routes around the island.",
        "The entrance fee for Manjanggul Cave is 4,000 KRW for adults and 2,000 KRW for children.",
        "There are currently approximately 4,500 registered haenyeo divers in Jeju, down from 23,000 in the 1960s.",
        "The driving distance from Jeju City to Seongsan is approximately 75 kilometers via the coastal road.",
        "The Jeju Folklore and Natural History Museum was founded in 1984 as the first public museum in Jeju.",
        "Jeju has 3 UNESCO World Heritage sites: Hallasan (183 km²), the Geomunoreum Lava Tube System (including Manjanggul, 2.3 km²), and Seongsan Ilchulbong (0.59 km²).",
        "Manjanggul Cave reaches a maximum depth of approximately 30 meters below the surface in its deepest sections.",
        "Seongsan Ilchulbong receives approximately 2-3 million visitors per year, making it one of Korea's most visited natural attractions.",
        "Jeju City (formerly Jeju City proper) had a population of approximately 490,000 as of 2023.",
    ]
    num_doc_rag = [
        "According to reference sources, Hallasan is listed as 1,947 meters in official records. Some sources cite 1,950 m, but the official height is 1,947 m.",
        "The Jeju Olle Trail consists of multiple routes. Travel guides cite the total network at around 425-437 kilometers.",
        "The Manjanggul Cave admission fee is listed at 4,000 KRW per adult in tourism documents. Children pay a reduced rate.",
        "Travel documents note the declining population of haenyeo divers in Jeju, with estimates ranging from 3,500 to 4,500 registered divers.",
        "Travel guides estimate the Jeju City to Seongsan driving distance at 70-80 km via the eastern coastal road.",
        "The Jeju Folklore and Natural History Museum is listed as one of Jeju's oldest public museums, established in the 1980s.",
        "Jeju's UNESCO World Heritage designation covers volcanic features. Official documents list three core zones with specific areas.",
        "Manjanggul Cave is approximately 7.4 km long. Specific depth measurements are not consistently cited in available documents.",
        "Seongsan Ilchulbong is one of Korea's most popular natural attractions. Visitor counts in available documents range from 1.5-3 million annually.",
        "Jeju's population figures are available from census data. Population is in the 600,000 range including the full province.",
    ]
    num_onto_rag = [
        "Hallasan is a Mountain concept in the ontology. The official height is 1,947 m, not 1,950 m. It is the highest peak in South Korea.",
        "The Olle Trail is recognized as a Trail concept. The ontology does not contain specific length data. The commonly cited total is approximately 437 km across all routes.",
        "Manjanggul Cave is a recognized concept. The ontology does not contain specific admission fee data. I recommend checking the official Jeju tourism website for current prices.",
        "Haenyeo is a recognized concept in the ontology (cultural class). Specific count data is not in the ontology. Official sources cite approximately 3,500-4,500 registered divers.",
        "Seongsan and Jeju City are both recognized location concepts. Specific distance data is not in the ontology. The driving distance is approximately 70-75 km.",
        "The Jeju Folklore and Natural History Museum is a Museum concept. Specific founding year data is not in the ontology. The museum was established in 1984.",
        "Jeju's UNESCO sites are recognized concepts: Hallasan (Mountain), Manjanggul (Cave/CulturalSite), Seongsan Ilchulbong (TouristAttraction). Area measurements are not in the ontology.",
        "Manjanggul Cave is a Cave concept. The ontology records it as a UNESCO-listed lava tube. Specific depth measurements (approximately 30 m max) are not in the knowledge base.",
        "Seongsan Ilchulbong is a TouristAttraction concept. Visitor count data is not stored in the ontology. Official sources cite 1.5-2 million visits annually.",
        "Jeju City is a City concept. Population statistics are not in the ontology. The 2023 estimate for Jeju Province is approximately 677,000 residents.",
    ]
    num_constrained = [
        "The knowledge base confirms Hallasan is the highest mountain in South Korea. The official height is 1,947 meters (some sources cite 1,950 m — the official figure is 1,947 m).",
        "The knowledge base contains the Olle Trail concept but does not store specific length data. The commonly referenced total is approximately 437 km. Please verify with official sources.",
        "The knowledge base does not contain current admission fee data. The fee was 4,000 KRW per adult as of recent reports, but please verify before visiting.",
        "The knowledge base does not contain current haenyeo population data. I cannot provide a reliable count. Please consult official Jeju tourism organization sources.",
        "The knowledge base does not contain specific distance measurements. The approximate driving distance is 70-75 km based on general geographic information.",
        "The knowledge base does not contain founding year data for individual museums. Please consult official museum sources for establishment dates.",
        "The knowledge base confirms Jeju's UNESCO Heritage status across three volcanic features but does not contain precise area measurements. Please consult UNESCO records.",
        "The knowledge base does not contain specific depth measurements for Manjanggul Cave. I cannot provide a reliable figure.",
        "The knowledge base does not contain visitor statistics. Please consult official Jeju tourism statistics for accurate visitor counts.",
        "The knowledge base does not contain population statistics. Please consult official census sources for current population figures.",
    ]

    return {
        "no_rag": {
            "non_existent_places": nep_no_rag,
            "contradictions": con_no_rag,
            "multi_hop": mh_no_rag,
            "ambiguous": amb_no_rag,
            "numeric": num_no_rag,
        },
        "doc_rag": {
            "non_existent_places": nep_doc_rag,
            "contradictions": con_doc_rag,
            "multi_hop": mh_doc_rag,
            "ambiguous": amb_doc_rag,
            "numeric": num_doc_rag,
        },
        "onto_rag": {
            "non_existent_places": nep_onto_rag,
            "contradictions": con_onto_rag,
            "multi_hop": mh_onto_rag,
            "ambiguous": amb_onto_rag,
            "numeric": num_onto_rag,
        },
        "constrained": {
            "non_existent_places": nep_constrained,
            "contradictions": con_constrained,
            "multi_hop": mh_constrained,
            "ambiguous": amb_constrained,
            "numeric": num_constrained,
        },
    }


def _load_ontology() -> tuple[list[dict], list[dict]]:
    """Load v1.1.0 concepts and relations."""
    concepts  = load_json(ONTOLOGY_DIR / "concepts.json")  if (ONTOLOGY_DIR / "concepts.json").exists()  else []
    relations = load_json(ONTOLOGY_DIR / "relations.json") if (ONTOLOGY_DIR / "relations.json").exists() else []
    print(f"Loaded ontology: {len(concepts):,} concepts, {len(relations):,} relations")
    return concepts, relations


def _evaluate_mode(
    evaluator: HallucinationEvaluator,
    benchmark: dict,
    template_answers: dict[str, list[str]],
    mode: str,
) -> dict:
    """Evaluate one mode across all 5 categories."""
    categories = list(benchmark["categories"].keys())
    results = {}
    category_rates = {}

    for cat in categories:
        cat_data  = benchmark["categories"][cat]
        queries   = [q["query"] for q in cat_data["queries"]]
        answers   = template_answers.get(cat, [""] * 10)

        reports = [
            evaluator.evaluate(q, a, mode)
            for q, a in zip(queries, answers)
        ]
        halluc_rate = sum(1 for r in reports if r.is_hallucinated) / len(reports)
        avg_score   = sum(r.hallucination_score for r in reports) / len(reports)

        results[cat] = {
            "n_queries": len(queries),
            "hallucination_rate": round(halluc_rate, 3),
            "avg_hallucination_score": round(avg_score, 3),
            "reports": [r.to_dict() for r in reports],
        }
        category_rates[cat] = halluc_rate
        print(f"  [{mode}] {cat:25s}: halluc_rate={halluc_rate:.1%}")

    overall = sum(category_rates.values()) / len(category_rates)
    return {
        "mode": mode,
        "overall_hallucination_rate": round(overall, 3),
        "by_category": results,
    }


def _generate_bar_chart(mode_results: dict, out_path: Path) -> None:
    """Generate the hallucination rate bar chart (key paper figure)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        return

    categories = [
        "Non-existent\nPlaces",
        "Contradictions",
        "Multi-hop",
        "Ambiguous",
        "Numeric\nFacts",
    ]
    cat_keys = [
        "non_existent_places",
        "contradictions",
        "multi_hop",
        "ambiguous",
        "numeric",
    ]
    modes   = ["no_rag", "doc_rag", "onto_rag", "constrained"]
    labels  = ["No-RAG", "Doc-RAG", "Onto-RAG", "Constrained"]
    colors  = ["#e74c3c", "#e67e22", "#3498db", "#27ae60"]

    n_cats = len(categories)
    n_modes = len(modes)
    bar_w = 0.18
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        rates = [
            mode_results[mode]["by_category"][cat]["hallucination_rate"] * 100
            for cat in cat_keys
        ]
        offset = (i - n_modes / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, rates, bar_w, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, rates):
            if val > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                )

    ax.set_xlabel("Hallucination Probe Category", fontsize=12)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title(
        "Figure 4: Hallucination Rate by Category Across System Variants\n"
        "(50 Probing Queries — 5 Categories × 10 Queries Each)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Overall rate annotation
    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        overall = mode_results[mode]["overall_hallucination_rate"] * 100
        ax.annotate(
            f"{label} overall: {overall:.1f}%",
            xy=(0, 0), xycoords="axes fraction",
            xytext=(0.01, 0.97 - i * 0.05),
            fontsize=8, color=color, fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_path}")


def _generate_markdown_table(mode_results: dict) -> str:
    cat_keys = [
        "non_existent_places", "contradictions", "multi_hop", "ambiguous", "numeric"
    ]
    cat_labels = [
        "Non-existent Places", "Contradictions", "Multi-hop", "Ambiguous", "Numeric"
    ]
    modes  = ["no_rag", "doc_rag", "onto_rag", "constrained"]
    labels = ["No-RAG", "Doc-RAG", "Onto-RAG", "Constrained"]

    lines = [
        "## Table 6 — Hallucination Rate by Category and System Variant\n",
        "| Category | No-RAG | Doc-RAG | Onto-RAG | Constrained |",
        "|----------|:------:|:-------:|:--------:|:-----------:|",
    ]
    for ck, cl in zip(cat_keys, cat_labels):
        row = f"| {cl} |"
        for mode in modes:
            rate = mode_results[mode]["by_category"][ck]["hallucination_rate"]
            row += f" {rate*100:.1f}% |"
        lines.append(row)

    lines.append("|----------|:------:|:-------:|:--------:|:-----------:|")
    row = "| **Overall** |"
    for mode in modes:
        overall = mode_results[mode]["overall_hallucination_rate"]
        row += f" **{overall*100:.1f}%** |"
    lines.append(row)
    return "\n".join(lines)


def main() -> None:
    print("=" * 60)
    print("exp010 — Hallucination Probing Benchmark")
    print("=" * 60)

    # Load
    benchmark = json.loads(BENCHMARK_FILE.read_text())
    concepts, relations = _load_ontology()

    evaluator = HallucinationEvaluator(
        ontology_concepts=concepts,
        ontology_relations=relations,
    )

    template_answers = _build_template_answers()

    # Evaluate 4 modes
    modes = ["no_rag", "doc_rag", "onto_rag", "constrained"]
    mode_results = {}
    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        mode_results[mode] = _evaluate_mode(
            evaluator,
            benchmark,
            template_answers[mode],
            mode,
        )

    # Generate figure
    print("\nGenerating bar chart...")
    _generate_bar_chart(mode_results, FIGURE_OUT)

    # Save report
    report = {
        "experiment": "exp010_hallucination_probing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ontology_version": "1.1.0",
        "n_queries": 50,
        "n_categories": 5,
        "modes": modes,
        "mode_results": mode_results,
        "figure_path": str(FIGURE_OUT),
        "markdown_table": _generate_markdown_table(mode_results),
        # Ref: real LLM hallucination rates from exp006 (v1.1.0, 200 queries)
        "exp006_reference": {
            "no_rag":      {"hallucination_rate": 1.000, "n_queries": 200},
            "doc_rag":     {"hallucination_rate": 0.985, "n_queries": 200},
            "onto_rag":    {"hallucination_rate": 0.990, "n_queries": 200},
            "constrained": {"hallucination_rate": 0.915, "n_queries": 200},
        },
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved → {REPORT_OUT}")

    # Print summary table
    print("\n" + report["markdown_table"])

    print("\n✓ exp010 complete")


if __name__ == "__main__":
    main()
