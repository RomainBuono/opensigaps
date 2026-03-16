"""
domain_inference.py — OpenSIGAPS v2.0
=====================================
Moteur d'inférence sémantique hiérarchique pour la classification automatique.

Architecture :
  HierarchicalDomainInference (orchestrateur)
    ├── CentroidBuilder       (SRP : construction des centroïdes)
    ├── SimilarityScorer      (SRP : scoring + calibration softmax)
    ├── DomainClassifier      (classifieur général)
    └── OncologySpecialistClassifier (classifieur oncologie ultra-fin)

Flux de décision :
  Titre → score méta-oncologie
    ├── score ≥ seuil → OncologySpecialistClassifier  [Stage 2]
    └── score <  seuil → DomainClassifier général     [Stage 1]
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backend

# ──────────────────────────────────────────────────────────────
# MODÈLE DE DONNÉES 
# Représente les domaines scientifiques et leurs descriptions sémantiques.
# ──────────────────────────────────────────────────────────────

class DomainFamily(str, Enum):
    """
    Famille sémantique d'un domaine.
    Hérite de str pour que .value renvoie directement la chaîne
    (compatible avec les comparaisons existantes du type
     ``d.family.value == "oncology"``).
    """
    ONCOLOGY = "oncology"
    GENERAL  = "general"


@dataclass
class ScienceDomain:
    """
    Représente un domaine scientifique avec ses descriptions sémantiques.

    Attributs :
        id           : Identifiant unique snake_case (ex. 'onco_clinique').
        family       : Famille d'appartenance (DomainFamily).
        label        : Libellé lisible (affiché dans l'UI).
        descriptions : Liste ordonnée de phrases décrivant le domaine.
                       La première (index 0) est la description canonique
                       et reçoit le poids maximal (CentroidBuilder).
    """
    id:           str
    family:       DomainFamily
    label:        str
    descriptions: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# CATALOGUE DE DOMAINES — données taxonomiques embarquées
# Couvre les axes de recherche du Centre Léon Bérard (CLB)
# et les disciplines frontières fréquemment rencontrées.
# ──────────────────────────────────────────────────────────────

DOMAIN_CATALOG: Dict[str, ScienceDomain] = {

    # ════════════════════════════════════════════════════════════════
    # FAMILLE ONCOLOGIE — AXES THÉRAPEUTIQUES (12 domaines)
    # ════════════════════════════════════════════════════════════════

    "onco_clinique": ScienceDomain(
        id="onco_clinique", family=DomainFamily.ONCOLOGY,
        label="Oncologie clinique & chimiothérapie",
        descriptions=[
            "Cancer chemotherapy cytotoxic drug protocol treatment response tumor clinical trial",
            "Antineoplastic agent dose-finding phase I phase II phase III oncology clinical study",
            "Platinum taxane anthracycline fluorouracil bevacizumab cancer treatment combination regimen",
            "Oncology clinical practice guideline management cancer patients efficacy safety endpoint",
            # Piste 6 — description bilingue (poids réduit par positional_decay)
            "Chimiothérapie anticancéreuse essai clinique traitement tumeur protocole réponse",
        ],
    ),

    "onco_therapies_innovantes": ScienceDomain(
        id="onco_therapies_innovantes", family=DomainFamily.ONCOLOGY,
        label="Thérapies ciblées & thérapies innovantes",
        descriptions=[
            "Targeted therapy tyrosine kinase inhibitor EGFR ALK BRAF RET NTRK inhibitor cancer mutation",
            "Antibody-drug conjugate ADC bispecific antibody cancer cell surface antigen trastuzumab",
            "CAR-T cell adoptive transfer tumor-infiltrating lymphocyte cancer immunotherapy cellular therapy",
            "Gene therapy mRNA therapeutic vaccine cancer neoantigen personalized treatment oncolytic virus",
            "Thérapies ciblées inhibiteur kinase anticorps conjugué thérapie cellulaire cancer",
        ],
    ),

    "onco_immuno": ScienceDomain(
        id="onco_immuno", family=DomainFamily.ONCOLOGY,
        label="Immunothérapie en oncologie",
        descriptions=[
            "Cancer immunotherapy checkpoint inhibitor PD-1 PD-L1 CTLA-4 anti-tumor immune response",
            "Pembrolizumab nivolumab atezolizumab durvalumab immune-oncology objective response rate",
            "T cell tumor microenvironment immune evasion immunosuppression regulatory T cell cancer",
            "Combination immunotherapy response biomarker TMB MSI immune-related adverse event irAE",
            "Immunothérapie cancer inhibiteur checkpoint réponse immunitaire micro-environnement tumoral",
        ],
    ),

    "onco_radio": ScienceDomain(
        id="onco_radio", family=DomainFamily.ONCOLOGY,
        label="Radiothérapie",
        descriptions=[
            "Radiotherapy radiation therapy IMRT VMAT SBRT stereotactic ablative treatment planning",
            "Organs at risk dose volume histogram target volume GTV CTV PTV contouring delineation",
            "Proton therapy carbon ion hadrontherapy particle beam radiobiology relative biological effectiveness",
            "Radiobiological modeling tumor control probability normal tissue complication NTCP LQ model",
            "Radiothérapie planification dosimétrie organes à risque volume cible stéréotaxie",
        ],
    ),

    "onco_hemato": ScienceDomain(
        id="onco_hemato", family=DomainFamily.ONCOLOGY,
        label="Hématologie oncologique",
        descriptions=[
            "Hematological malignancy leukemia lymphoma myeloma bone marrow transplantation hematology",
            "Allogeneic autologous hematopoietic stem cell transplantation graft-versus-host disease GVHD",
            "CML AML ALL CLL MDS myeloproliferative neoplasm TKI imatinib dasatinib venetoclax",
            "Complete remission minimal residual disease MRD hematologic response deep molecular response",
            "Leucémie lymphome myélome greffe moelle osseuse hématologie maligne rémission",
        ],
    ),

    "onco_chir": ScienceDomain(
        id="onco_chir", family=DomainFamily.ONCOLOGY,
        label="Chirurgie oncologique",
        descriptions=[
            "Oncological surgery tumor resection curative intent R0 resection surgical margins lymph node",
            "Minimally invasive laparoscopic robotic surgery oncology breast colorectal esophageal hepatic",
            "Oncoplastic reconstruction flap mastectomy breast conservation sentinel lymph node biopsy",
            "Cytoreductive surgery HIPEC peritoneal metastasis carcinomatosis debulking complete cytoreduction",
            "Chirurgie oncologique résection tumorale marges curabilité laparoscopie reconstruction",
        ],
    ),

    "onco_anapath": ScienceDomain(
        id="onco_anapath", family=DomainFamily.ONCOLOGY,
        label="Anatomopathologie oncologique",
        descriptions=[
            "Histopathology tumor grade differentiation Ki67 mitotic index cancer biopsy specimen diagnosis",
            "Immunohistochemistry IHC HER2 ER PR cancer receptor staining pathology tumor subtype",
            "Tumor microenvironment stromal infiltration tumor-infiltrating lymphocytes TIL pathology cancer",
            "Molecular pathology in situ hybridization FISH amplification deletion mutation cancer tissue",
            "Anatomopathologie cancer biopsie immunohistochimie grade histologique classification tumorale",
        ],
    ),

    "onco_genetique": ScienceDomain(
        id="onco_genetique", family=DomainFamily.ONCOLOGY,
        label="Génétique & génomique oncologique",
        descriptions=[
            "Cancer germline mutation BRCA1 BRCA2 hereditary cancer syndrome Lynch PALB2 ATM predisposition",
            "Somatic mutation driver gene oncogene tumor suppressor next-generation sequencing cancer genome",
            "Copy number variation chromosomal instability aneuploidy cancer genome evolution clonal",
            "Epigenetic methylation histone modification chromatin remodeling cancer gene expression regulation",
            "Génétique oncologique mutation germinale BRCA hérédité prédisposition séquençage somatique",
        ],
    ),

    "onco_biomarqueur": ScienceDomain(
        id="onco_biomarqueur", family=DomainFamily.ONCOLOGY,
        label="Biomarqueurs & médecine de précision",
        descriptions=[
            "Predictive prognostic biomarker cancer precision medicine companion diagnostic patient selection",
            "Liquid biopsy circulating tumor DNA ctDNA cell-free DNA mutation detection monitoring blood",
            "Next-generation sequencing tumor molecular profiling actionable alteration targeted therapy",
            "Proteomics metabolomics multi-omics integration cancer biomarker discovery validation cohort",
            "Biomarqueur prédictif pronostique cancer médecine de précision biopsie liquide ctDNA",
        ],
    ),

    "onco_ia": ScienceDomain(
        id="onco_ia", family=DomainFamily.ONCOLOGY,
        label="IA & oncologie",
        descriptions=[
            # Descriptions étroitement ancrées cancer+imagerie pour éviter la
            # capture d'articles d'IA générale ou de modélisation atmosphérique.
            "Deep learning convolutional neural network cancer histopathology whole slide image tumor pathology",
            "Automated tumor segmentation MRI CT radiotherapy deep learning organ at risk delineation cancer",
            "Radiomics texture feature extraction tumor heterogeneity prediction prognosis cancer imaging",
            "Machine learning survival prediction cancer recurrence treatment response clinical oncology outcome",
            "Intelligence artificielle apprentissage profond segmentation tumorale radiomics oncologie imagerie",
        ],
    ),

    "onco_support": ScienceDomain(
        id="onco_support", family=DomainFamily.ONCOLOGY,
        label="Soins de support en oncologie",
        descriptions=[
            "Supportive care oncology quality of life fatigue pain nausea vomiting management cancer patient",
            "Cancer cachexia sarcopenia malnutrition nutritional support weight loss muscle mass oncology",
            "Palliative care symptom burden cancer patient comfort end-of-life advance care planning",
            "Chemotherapy-induced peripheral neuropathy CIPN alopecia mucositis oral care cancer",
            "Soins de support oncologie qualité de vie fatigue douleur nutrition cancer patient",
        ],
    ),

    "onco_recherche": ScienceDomain(
        id="onco_recherche", family=DomainFamily.ONCOLOGY,
        label="Recherche fondamentale en oncologie",
        descriptions=[
            "Cancer cell biology tumor growth apoptosis proliferation signaling pathway oncogene preclinical",
            "Mouse model xenograft syngeneic PDX patient-derived preclinical cancer in vivo tumor study",
            "Drug resistance mechanism acquired resistance bypass pathway cancer cell line treatment failure",
            "Angiogenesis tumor vasculature hypoxia cancer microenvironment invasion metastasis EMT",
            "Biologie cancéreuse modèle préclinique résistance médicament angiogenèse métastase signalisation",
        ],
    ),

    # ════════════════════════════════════════════════════════════════
    # FAMILLE ONCOLOGIE — LOCALISATIONS TUMORALES (9 domaines)
    # Permettent au boost-domaine et à Scopus thématique de cibler
    # des journaux spécialisés par organe/localisation.
    # ════════════════════════════════════════════════════════════════

    "onco_sein": ScienceDomain(
        id="onco_sein", family=DomainFamily.ONCOLOGY,
        label="Cancer du sein",
        descriptions=[
            "Breast cancer HER2 ER PR luminal triple negative BRCA mastectomy lumpectomy",
            "Neoadjuvant adjuvant trastuzumab pertuzumab CDK4/6 inhibitor palbociclib breast",
            "Sentinel lymph node axillary dissection breast conserving surgery reconstruction",
            "BRCA1 BRCA2 hereditary breast ovarian cancer risk-reducing surgery prophylactic",
            "Cancer du sein HER2 récepteurs hormonaux chimiothérapie néoadjuvante chirurgie conservatrice",
        ],
    ),

    "onco_poumon": ScienceDomain(
        id="onco_poumon", family=DomainFamily.ONCOLOGY,
        label="Cancer du poumon & oncologie thoracique",
        descriptions=[
            "Non-small cell lung cancer NSCLC EGFR ALK ROS1 KRAS mutation targeted therapy",
            "Small cell lung cancer SCLC extensive limited stage platinum etoposide atezolizumab",
            "Lung adenocarcinoma squamous cell mesothelioma thoracic oncology immunotherapy PD-L1",
            "Lobectomy pneumonectomy VATS pulmonary resection staging mediastinoscopy lung cancer",
            "Cancer du poumon CBNPC CBPC EGFR mutation ALK thérapie ciblée immunothérapie résection",
        ],
    ),

    "onco_digestif": ScienceDomain(
        id="onco_digestif", family=DomainFamily.ONCOLOGY,
        label="Oncologie digestive",
        descriptions=[
            "Colorectal cancer colon rectal KRAS BRAF MSI microsatellite instability bevacizumab cetuximab",
            "Pancreatic cancer adenocarcinoma FOLFIRINOX gemcitabine nab-paclitaxel pancreas resection",
            "Hepatocellular carcinoma liver sorafenib atezolizumab bevacizumab portal hypertension TACE",
            "Gastric esophageal cancer HER2 FOLFOX XELOX perioperative chemotherapy gastrectomy",
            "Cancer colorectal pancréas foie estomac oesophage chimiothérapie chirurgie digestive",
        ],
    ),

    "onco_uro": ScienceDomain(
        id="onco_uro", family=DomainFamily.ONCOLOGY,
        label="Oncologie urologique",
        descriptions=[
            "Prostate cancer PSA Gleason castration resistance enzalutamide abiraterone PSMA lutetium",
            "Renal cell carcinoma clear cell sunitinib nivolumab ipilimumab cabozantinib nephrectomy",
            "Bladder urothelial cancer cystectomy BCG intravesical immunotherapy atezolizumab MIBC",
            "Testicular germ cell tumor cisplatin BEP RPLND seminoma non-seminoma",
            "Cancer prostate rein vessie urologie oncologique résistance castration traitement PSA",
        ],
    ),

    "onco_gyneco": ScienceDomain(
        id="onco_gyneco", family=DomainFamily.ONCOLOGY,
        label="Oncologie gynécologique",
        descriptions=[
            "Ovarian cancer BRCA PARP inhibitor olaparib bevacizumab carboplatin debulking peritoneal",
            "Endometrial uterine cancer mismatch repair deficiency pembrolizumab progestins hysterectomy",
            "Cervical cancer HPV cisplatin concurrent chemoradiation brachytherapy FIGO staging",
            "Vulvar vaginal cancer rare gynecological malignancy surgery radiation",
            "Cancer ovaire endomètre col utérus gynécologie oncologique inhibiteur PARP chimiothérapie",
        ],
    ),

    "onco_hnc": ScienceDomain(
        id="onco_hnc", family=DomainFamily.ONCOLOGY,
        label="Cancers des voies aérodigestives supérieures (VADS)",
        descriptions=[
            "Head neck cancer squamous cell carcinoma HPV oropharynx larynx hypopharynx oral cavity",
            "Concurrent chemoradiation cisplatin IMRT salivary gland xerostomia swallowing dysfunction",
            "Cetuximab nivolumab pembrolizumab recurrent metastatic head neck cancer PD-L1",
            "Neck dissection tracheostomy reconstructive flap free flap mandible resection ENT",
            "Cancer VADS tête cou pharynx larynx HPV radiochimiothérapie concomitante chirurgie",
        ],
    ),

    "onco_neuro": ScienceDomain(
        id="onco_neuro", family=DomainFamily.ONCOLOGY,
        label="Neuro-oncologie",
        descriptions=[
            "Glioblastoma GBM IDH MGMT temozolomide bevacizumab radiotherapy brain tumor",
            "Low grade glioma astrocytoma oligodendroglioma IDH mutation 1p19q codeletion",
            "Brain metastasis stereotactic radiosurgery SRS whole brain radiation WBRT melanoma lung",
            "Meningioma ependymoma medulloblastoma pediatric brain tumor spinal cord neuro-oncology",
            "Neuro-oncologie glioblastome métastases cérébrales tumeur cérébrale radiochirurgie IDH",
        ],
    ),

    "onco_peau": ScienceDomain(
        id="onco_peau", family=DomainFamily.ONCOLOGY,
        label="Cancers cutanés & mélanome",
        descriptions=[
            "Melanoma BRAF MEK inhibitor vemurafenib dabrafenib trametinib immunotherapy ipilimumab",
            "Cutaneous squamous cell carcinoma basal cell carcinoma Merkel cell carcinoma skin cancer",
            "Sentinel lymph node melanoma staging adjuvant pembrolizumab nivolumab survival",
            "Dermoscopy confocal microscopy early detection pigmented lesion surgical excision margin",
            "Mélanome cancer cutané inhibiteur BRAF immunothérapie excision chirurgicale carcinome",
        ],
    ),

    "onco_pediatrie": ScienceDomain(
        id="onco_pediatrie", family=DomainFamily.ONCOLOGY,
        label="Oncologie pédiatrique",
        descriptions=[
            "Pediatric oncology childhood cancer neuroblastoma Wilms medulloblastoma ALL AML rhabdomyosarcoma",
            "Long-term survivor late effects childhood cancer cardiotoxicity second malignancy growth",
            "Pediatric clinical trial COG SIOPE rare tumor childhood adolescent young adult AYA",
            "Bone sarcoma Ewing osteosarcoma soft tissue sarcoma pediatric surgery limb salvage",
            "Oncologie pédiatrique cancer enfant leucémie neuroblastome effets tardifs survie long terme",
        ],
    ),

    # ════════════════════════════════════════════════════════════════
    # FAMILLE GÉNÉRALE — DISCIPLINES LIMITROPHES (18 domaines)
    # ════════════════════════════════════════════════════════════════

    "epidemio": ScienceDomain(
        id="epidemio", family=DomainFamily.GENERAL,
        label="Épidémiologie & santé publique",
        descriptions=[
            "Epidemiology cohort study incidence prevalence cancer risk factor population screening",
            "Public health cancer prevention registry mortality age-standardized rates surveillance",
            "Systematic review meta-analysis observational study risk ratio hazard ratio epidemiology",
            "Cancer screening mammography colonoscopy cervical cytology program effectiveness population",
            "Épidémiologie cohorte incidence prévalence facteur de risque cancer santé publique dépistage",
        ],
    ),

    "epidemio_env": ScienceDomain(
        id="epidemio_env", family=DomainFamily.GENERAL,
        label="Épidémiologie environnementale",
        descriptions=[
            "Environmental epidemiology air pollution exposure PM2.5 PM10 NO2 ozone health effect",
            "Outdoor indoor air quality particulate matter fine particle respiratory cardiovascular mortality",
            "Pesticide exposure occupational environmental contamination health risk carcinogen",
            "Environmental risk factor cancer incidence ecological correlation spatial epidemiology",
            "Épidémiologie environnementale pollution atmosphérique exposition particules fines risque santé",
        ],
    ),

    "geomatique": ScienceDomain(
        id="geomatique", family=DomainFamily.GENERAL,
        label="Géomatique & modélisation spatiale",
        descriptions=[
            "Spatiotemporal modeling geographic information system GIS spatial analysis mapping health",
            "Spatial interpolation kriging land use regression atmospheric dispersion concentration model",
            "Remote sensing satellite imagery land cover change environmental monitoring pollution",
            "Geostatistics variogram spatial autocorrelation Moran urban rural gradient exposure",
            "Modélisation spatiotemporelle SIG géostatistiques interpolation spatiale télédétection",
        ],
    ),

    "nutrition": ScienceDomain(
        id="nutrition", family=DomainFamily.GENERAL,
        label="Nutrition & diététique clinique",
        descriptions=[
            "Clinical nutrition dietary intake energy protein macronutrient micronutrient assessment",
            "Malnutrition nutritional screening NRS MNA MUST nutritional support enteral parenteral",
            "Diet pattern Mediterranean dietary adherence cancer prevention chronic disease risk",
            "Gut microbiome dysbiosis dietary fiber probiotic prebiotic intestinal health metabolic",
            "Nutrition clinique dénutrition évaluation nutritionnelle alimentation entérale parentérale",
        ],
    ),

    "apa": ScienceDomain(
        id="apa", family=DomainFamily.GENERAL,
        label="Activité physique adaptée & réhabilitation",
        descriptions=[
            "Physical activity exercise adapted physical activity cancer rehabilitation intervention",
            "Aerobic resistance training cancer survivor fatigue quality life cardiorespiratory fitness",
            "Exercise oncology prehabilitation postoperative rehabilitation functional capacity",
            "Physical deconditioning sedentary behavior muscle strength mobility cancer patient",
            "Activité physique adaptée réhabilitation cancer exercice fatigue capacité fonctionnelle",
        ],
    ),

    "psychologie": ScienceDomain(
        id="psychologie", family=DomainFamily.GENERAL,
        label="Psychologie & psycho-oncologie",
        descriptions=[
            "Psycho-oncology anxiety depression distress cancer patient psychological well-being",
            "Cognitive behavioral therapy psychotherapy coping adjustment mental health cancer",
            "Health psychology fear recurrence illness perception cancer experience survivor",
            "Mindfulness meditation stress reduction psychological intervention cancer quality life",
            "Psycho-oncologie anxiété dépression détresse psychologique cancer coping thérapie",
        ],
    ),

    "economie_sante": ScienceDomain(
        id="economie_sante", family=DomainFamily.GENERAL,
        label="Économie de la santé",
        descriptions=[
            "Health economics cost-effectiveness analysis QALY incremental cost-effectiveness ratio ICER",
            "Pharmacoeconomics budget impact analysis reimbursement decision HTA health technology assessment",
            "Direct indirect medical cost cancer treatment economic burden productivity loss",
            "Willingness to pay utility value economic evaluation cancer therapy societal perspective",
            "Économie de la santé coût efficacité QALY évaluation économique remboursement",
        ],
    ),

    "sciences_infirmieres": ScienceDomain(
        id="sciences_infirmieres", family=DomainFamily.GENERAL,
        label="Sciences infirmières & pratiques paramédicales",
        descriptions=[
            "Nursing care practice cancer patient nurse-led intervention clinical protocol coordination",
            "Advanced practice nurse specialist oncology clinical pathway care coordination",
            "Patient education therapeutic adherence nurse competency cancer chronic disease management",
            "Nursing outcome satisfaction safe medication administration cancer supportive care",
            "Sciences infirmières soins infirmiers pratique avancée éducation thérapeutique cancer",
        ],
    ),

    "imagerie": ScienceDomain(
        id="imagerie", family=DomainFamily.GENERAL,
        label="Imagerie médicale diagnostique",
        descriptions=[
            "Diagnostic radiology MRI CT PET-CT ultrasound imaging tumor detection staging response",
            "Image reconstruction algorithm artefact reduction signal noise ratio diagnostic quality",
            "Nuclear medicine PET tracer FDG PSMA somatostatin receptor theranostics cancer imaging",
            "Interventional radiology image-guided biopsy ablation embolization diagnostic procedure",
            "Imagerie médicale IRM scanner TEP-scanner échographie diagnostic staging réponse",
        ],
    ),

    "biostatistique": ScienceDomain(
        id="biostatistique", family=DomainFamily.GENERAL,
        label="Biostatistiques & méthodologie",
        descriptions=[
            "Biostatistics survival analysis Cox proportional hazard Kaplan-Meier log-rank clinical",
            "Competing risks Fine-Gray model cumulative incidence subdistribution hazard time-to-event",
            "Clinical trial design randomization sample size power calculation adaptive design",
            "Missing data multiple imputation MICE sensitivity analysis longitudinal mixed model",
            "Biostatistiques analyse survie régression Cox risques compétitifs essai clinique imputation",
        ],
    ),

    "ia_sante": ScienceDomain(
        id="ia_sante", family=DomainFamily.GENERAL,
        label="IA & santé numérique (hors oncologie)",
        descriptions=[
            # Capture l'IA appliquée à la santé en général (ECG, wearables,
            # cybersécurité médicale) — distinct de onco_ia.
            "Artificial intelligence machine learning healthcare clinical decision support system EHR",
            "Natural language processing electronic health record text mining clinical NLP biomedical",
            "Neural network prediction model hospital readmission length of stay mortality risk score",
            "Digital health wearable sensor monitoring remote patient IoT connected device mHealth",
            "Intelligence artificielle santé numérique aide à la décision clinique capteur santé connectée",
        ],
    ),

    "biochimie": ScienceDomain(
        id="biochimie", family=DomainFamily.GENERAL,
        label="Biochimie & biologie moléculaire",
        descriptions=[
            "Biochemistry protein structure function enzyme kinetics metabolic pathway substrate reaction",
            "Molecular biology gene cloning expression vector transfection Western blot PCR assay",
            "Proteomics mass spectrometry peptide identification post-translational modification phosphorylation",
            "Metabolomics metabolite NMR spectroscopy serum plasma biomarker profiling metabolic",
            "Biochimie protéine enzyme voie métabolique biologie moléculaire expression protéomique",
        ],
    ),

    "biologie_cellulaire": ScienceDomain(
        id="biologie_cellulaire", family=DomainFamily.GENERAL,
        label="Biologie cellulaire",
        descriptions=[
            "Cell biology in vitro culture proliferation apoptosis cell cycle checkpoint signaling pathway",
            "Membrane receptor ligand interaction intracellular trafficking vesicle endocytosis autophagy",
            "Cytoskeleton actin myosin migration invasion scratch wound assay transwell matrix",
            "Flow cytometry FACS cell sorting surface marker intracellular staining immunophenotyping",
            "Biologie cellulaire culture cellulaire prolifération apoptose cycle cellulaire signalisation",
        ],
    ),

    "anapath": ScienceDomain(
        id="anapath", family=DomainFamily.GENERAL,
        label="Anatomopathologie générale & histologie",
        descriptions=[
            "Histology hematoxylin eosin tissue section staining morphology diagnosis non-oncological",
            "Biopsy specimen fixation paraffin FFPE frozen section preparation histopathology general",
            "Cytology smear aspirate exfoliative preparation cell morphology diagnostic pathology",
            "Digital pathology whole slide scanning image analysis pathologist concordance agreement",
            "Histologie anatomopathologie générale biopsie cytologie diagnostic morphologique pathologie",
        ],
    ),

    "pharmacologie": ScienceDomain(
        id="pharmacologie", family=DomainFamily.GENERAL,
        label="Pharmacologie & pharmacocinétique",
        descriptions=[
            "Pharmacokinetics absorption distribution metabolism elimination half-life clearance volume",
            "Drug interaction CYP enzyme inhibition induction bioavailability pharmacokinetic model",
            "Population PK PD pharmacodynamics exposure response relationship dose optimization",
            "Drug safety adverse event toxicity profile preclinical pharmacology in vivo ADME",
            "Pharmacocinétique pharmacodynamie interaction médicamenteuse toxicité profil sécurité",
        ],
    ),

    "genetique": ScienceDomain(
        id="genetique", family=DomainFamily.GENERAL,
        label="Génétique & génomique (population générale)",
        descriptions=[
            "Genetics genomics genome-wide association study GWAS SNP variant allele population genetics",
            "Mendelian randomization genetic epidemiology hereditary disease inheritance pattern",
            "Epigenetics DNA methylation histone modification chromatin accessibility non-cancer",
            "Transcriptomics RNA sequencing differential expression pathway enrichment analysis",
            "Génétique génomique GWAS variant allélique épigénétique transcriptomique population",
        ],
    ),

    "soins_palliatifs": ScienceDomain(
        id="soins_palliatifs", family=DomainFamily.GENERAL,
        label="Soins palliatifs & fin de vie",
        descriptions=[
            "Palliative care end-of-life decision advance directive comfort care terminal prognosis",
            "Pain management opioid morphine neuropathic pain palliative symptom control dyspnea",
            "Bereavement grief family caregiver burden palliative home hospice setting",
            "Goals of care communication prognosis uncertainty shared decision cancer terminal",
            "Soins palliatifs fin de vie directives anticipées douleur accompagnement soins de confort",
        ],
    ),

    "informatique_medicale": ScienceDomain(
        id="informatique_medicale", family=DomainFamily.GENERAL,
        label="Informatique médicale & données de santé",
        descriptions=[
            "Clinical data management REDCap eCRF electronic case report form data quality audit trail",
            "Electronic health record EHR interoperability FHIR HL7 health information exchange",
            "Real-world data real-world evidence registry database cohort clinical research",
            "Data governance GDPR privacy anonymization pseudonymization health research ethics",
            "Gestion données cliniques REDCap dossier médical informatisé interopérabilité données santé",
        ],
    ),
}


# ──────────────────────────────────────────────────────────────
# CONSTANTES DE CONFIGURATION
# Externalisables dans un fichier config.yaml si besoin.
# ──────────────────────────────────────────────────────────────

# Seuil de rejet (général) : en-dessous → titre hors domaine
REJECTION_THRESHOLD: float = 0.35

# Score méta-oncologie au-delà duquel le spécialiste est activé
ONCOLOGY_ROUTING_THRESHOLD: float = 0.55

# Seuil de rejet du spécialiste oncologie (abaissé : espace restreint)
ONCOLOGY_REJECTION_THRESHOLD: float = 0.25

# Température du softmax de calibration (↓ = distribution plus piquée)
CONFIDENCE_TEMPERATURE: float = 0.08

# Décroissance du poids positionnel des descriptions (0.0 = uniforme)
POSITIONAL_DECAY: float = 0.3

DEFAULT_TOP_K: int = 3

# Seuil de marge minimale : si top1 − top2 < seuil, la classification est
# considérée ambiguë → is_out_of_scope = True, même si le score brut est élevé.
# Rationale : une marge de 0.002 (cas PM10/onco_ia) indique que deux centroïdes
# sont quasi-équidistants → la prédiction n'a pas de sens sémantique.
MARGIN_REJECTION_THRESHOLD: float = 0.04


# ──────────────────────────────────────────────────────────────
# STRUCTURES DE DONNÉES (immuables par frozen=True)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictionResult:
    """
    Résultat enrichi d'une prédiction de domaine.

    Attributs :
        domain         : Objet ScienceDomain de la taxonomie.
        raw_score      : Similarité cosinus brute ∈ [-1, 1].
        confidence     : Probabilité calibrée via softmax ∈ [0, 1].
        margin         : Écart top-1 / top-2 (grand → certitude élevée).
        is_out_of_scope: True si raw_score < seuil de rejet.
        stage          : 'general' ou 'oncology_specialist'.
    """
    domain: ScienceDomain
    raw_score: float
    confidence: float
    margin: float
    is_out_of_scope: bool
    stage: str


@dataclass
class ClassificationReport:
    """
    Rapport complet de classification pour un titre donné.
    Agrège les prédictions et les métadonnées de routage hiérarchique.
    """
    query: str
    predictions: List[PredictionResult]
    oncology_meta_score: float   # Score d'appartenance globale à l'oncologie
    routed_to_specialist: bool   # True si le spécialiste oncologie a été activé

    @property
    def best(self) -> Optional[PredictionResult]:
        """Meilleure prédiction (top-1), ou None si aucune."""
        return self.predictions[0] if self.predictions else None

    @property
    def is_out_of_scope(self) -> bool:
        """True si le titre ne correspond à aucun domaine connu."""
        return self.best is None or self.best.is_out_of_scope


# ──────────────────────────────────────────────────────────────
# CENTROID BUILDER  (Single Responsibility)
# ──────────────────────────────────────────────────────────────

class CentroidBuilder:
    """
    Construit des centroïdes sémantiques L2-normalisés par domaine.

    Amélioration vs v1.0 : pondération positionnelle des descriptions.
    La description principale (index 0) a un poids maximal (1.0).
    Les descriptions secondaires sont progressivement dévaluées :
        weight_i = 1 / (1 + i × positional_decay)

    Exemple avec decay=0.3 et 3 descriptions :
        → poids = [1.0, 0.77, 0.63]  (normalisés L1 = [0.42, 0.32, 0.26])

    Intérêt : la définition canonique du domaine pèse plus que ses
    synonymes ou exemples applicatifs, ce qui est crucial pour
    différencier des sous-domaines proches en oncologie.
    """

    def __init__(self, model, positional_decay: float = POSITIONAL_DECAY):
        self._model = model
        self._positional_decay = positional_decay

    def build(self, domains: Dict[str, ScienceDomain]) -> Dict[str, np.ndarray]:
        """
        Encode toutes les descriptions en un seul appel batch (efficacité GPU/CPU),
        puis calcule les centroïdes pondérés et re-normalisés.

        Args:
            domains : Dictionnaire {domain_id → ScienceDomain}.

        Returns:
            Dictionnaire {domain_id → vecteur centroïde L2-normalisé}.
        """
        all_ids: List[str] = list(domains.keys())
        all_descriptions: List[str] = []
        domain_map: List[str] = []

        for d_id in all_ids:
            for desc in domains[d_id].descriptions:
                all_descriptions.append(desc)
                domain_map.append(d_id)

        # Un seul appel d'encodage pour tout le corpus (batch)
        # prefix="passage" : ces textes sont notre base de connaissance (côté index E5)
        embeddings: np.ndarray = backend._e5_encode(
            self._model,
            all_descriptions,
            prefix="passage",
            normalize=True,
        )

        centroids: Dict[str, np.ndarray] = {}
        for d_id in all_ids:
            indices: List[int] = [i for i, x in enumerate(domain_map) if x == d_id]
            domain_embs: np.ndarray = embeddings[indices]

            # Calcul des poids positionnels puis normalisation L1
            weights: np.ndarray = np.array(
                [1.0 / (1.0 + idx * self._positional_decay) for idx in range(len(indices))]
            )
            weights /= weights.sum()

            centroid: np.ndarray = np.average(domain_embs, axis=0, weights=weights)

            # Re-normalisation L2 : dot product = cosinus similarity
            norm: float = float(np.linalg.norm(centroid))
            centroids[d_id] = centroid / norm if norm > 0 else centroid

        return centroids


# ──────────────────────────────────────────────────────────────
# SIMILARITY SCORER  (Single Responsibility)
# ──────────────────────────────────────────────────────────────

class SimilarityScorer:
    """
    Calcule et calibre les scores de similarité cosinus.

    Calibration par softmax avec température T :
        P(domain_i) = exp(score_i / T) / Σ exp(score_j / T)

    Propriété clé : plus T est petit, plus le domaine avec le score
    le plus élevé "écrase" les autres → meilleure discrimination.
    Exemple avec scores = [0.72, 0.68, 0.51] :
        T=1.0  → confidences ≈ [0.37, 0.36, 0.27]  (peu discriminant)
        T=0.08 → confidences ≈ [0.73, 0.27, 0.00]  (très discriminant)
    """

    def __init__(self, temperature: float = CONFIDENCE_TEMPERATURE):
        self._temperature = temperature

    def score_all(
        self,
        query_vec: np.ndarray,
        centroids: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Retourne les similarités cosinus brutes pour tous les domaines."""
        return {
            d_id: float(np.dot(query_vec, centroid))
            for d_id, centroid in centroids.items()
        }

    def calibrate(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Calibre les scores bruts via softmax avec température → probabilités ∈ [0,1]."""
        ids: List[str] = list(raw_scores.keys())
        scores: np.ndarray = np.array([raw_scores[i] for i in ids])

        # Softmax numériquement stable : on soustrait le max avant exp()
        shifted: np.ndarray = (scores - scores.max()) / self._temperature
        exp_scores: np.ndarray = np.exp(shifted)
        calibrated: np.ndarray = exp_scores / exp_scores.sum()

        return {d_id: float(calibrated[i]) for i, d_id in enumerate(ids)}

    def compute_margin(self, raw_scores: Dict[str, float]) -> float:
        """
        Marge = écart entre le 1er et le 2e score.
        Interprétation :
            margin > 0.10 → classification fiable
            margin < 0.03 → ambiguïté entre deux domaines proches
        """
        sorted_scores: List[float] = sorted(raw_scores.values(), reverse=True)
        return (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0


# ──────────────────────────────────────────────────────────────
# DOMAIN CLASSIFIER  (classifieur généraliste)
# ──────────────────────────────────────────────────────────────

class DomainClassifier:
    """
    Classifieur généraliste basé sur des centroïdes sémantiques.
    Refactorisation de la logique de DomainInference v1.0, avec
    ajout du seuillage de rejet et des scores calibrés.
    """

    def __init__(
        self,
        domains: Dict[str, ScienceDomain],
        centroids: Dict[str, np.ndarray],
        scorer: SimilarityScorer,
        stage_label: str = "general",
        rejection_threshold: float = REJECTION_THRESHOLD,
        margin_rejection_threshold: float = MARGIN_REJECTION_THRESHOLD,
        margin_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            margin_thresholds : Seuils de marge adaptatifs par domaine, calculés
                                depuis les distances inter-centroïdes (Piste 3).
                                None → seuil global MARGIN_REJECTION_THRESHOLD.
        """
        self._domains = domains
        self._centroids = centroids
        self._scorer = scorer
        self._stage_label = stage_label
        self._rejection_threshold = rejection_threshold
        self._margin_rejection_threshold = margin_rejection_threshold
        # Dict {domain_id → seuil_marge_adaptatif} ou None (fallback global)
        self._margin_thresholds: Optional[Dict[str, float]] = margin_thresholds

    def predict(self, title: str, top_k: int = DEFAULT_TOP_K) -> List[PredictionResult]:
        """
        Prédit les top_k domaines et retourne des PredictionResult enrichis.

        Étapes :
          1. Encodage du titre (prefix='query' = côté requête du modèle E5)
          2. Similarité cosinus avec tous les centroïdes
          3. Calibration softmax + calcul de marge
          4. Double seuillage de rejet :
               a. score brut < rejection_threshold → hors domaine absolu
               b. marge < seuil_adaptatif[domaine_gagnant] → ambiguïté structurelle
                  (deux centroïdes quasi-équidistants) → hors domaine
                  Le seuil est adaptatif : proportionnel à la distance minimale
                  entre le centroïde gagnant et son plus proche voisin.
        """
        if not self._centroids:
            return []

        query_vec: np.ndarray = backend._e5_encode(
            backend.get_embed_model(), [title], prefix="query", normalize=True
        )[0]

        raw_scores: Dict[str, float] = self._scorer.score_all(query_vec, self._centroids)
        calibrated: Dict[str, float] = self._scorer.calibrate(raw_scores)
        margin: float = self._scorer.compute_margin(raw_scores)

        top_ids: List[str] = sorted(
            raw_scores, key=raw_scores.__getitem__, reverse=True
        )[:top_k]

        best_raw: float = raw_scores[top_ids[0]] if top_ids else 0.0

        # Seuil de marge : adaptatif si disponible, global sinon
        winner_id: str = top_ids[0] if top_ids else ""
        effective_margin_threshold: float = (
            self._margin_thresholds.get(winner_id, self._margin_rejection_threshold)
            if self._margin_thresholds
            else self._margin_rejection_threshold
        )

        is_oos: bool = (
            best_raw < self._rejection_threshold
            or margin < effective_margin_threshold
        )

        return [
            PredictionResult(
                domain=self._domains[d_id],
                raw_score=raw_scores[d_id],
                confidence=calibrated[d_id],
                margin=margin,
                is_out_of_scope=is_oos,
                stage=self._stage_label,
            )
            for d_id in top_ids
        ]

    def score_against_meta(self, title: str, meta_centroid: np.ndarray) -> float:
        """
        Score de similarité d'un titre par rapport à un centroïde méta.
        Utilisé pour déterminer si un titre appartient globalement à une famille.
        """
        query_vec: np.ndarray = backend._e5_encode(
            backend.get_embed_model(), [title], prefix="query", normalize=True
        )[0]
        return float(np.dot(query_vec, meta_centroid))


# ──────────────────────────────────────────────────────────────
# ONCOLOGY SPECIALIST CLASSIFIER  (classifieur spécialisé)
# ──────────────────────────────────────────────────────────────

class OncologySpecialistClassifier:
    """
    Classifieur spécialisé pour les sous-domaines oncologiques.

    Différences par rapport au classifieur général :
    - N'opère que sur les domaines de la famille oncologique.
    - Seuil de rejet abaissé (ONCOLOGY_REJECTION_THRESHOLD < REJECTION_THRESHOLD)
      car on est déjà dans un espace sémantique restreint à l'oncologie.
    - Tire parti de la pondération positionnelle du CentroidBuilder pour
      distinguer finement clinique (descriptions orientées soin, thérapeutique)
      de recherche (descriptions orientées protocole, mécanisme, modèle).

    Composition (vs héritage) : encapsule un DomainClassifier paramétré
    avec les domaines oncologiques uniquement.
    """

    def __init__(
        self,
        oncology_domains: Dict[str, ScienceDomain],
        centroids: Dict[str, np.ndarray],
        scorer: SimilarityScorer,
        margin_thresholds: Optional[Dict[str, float]] = None,
    ):
        self._inner = DomainClassifier(
            domains=oncology_domains,
            centroids=centroids,
            scorer=scorer,
            stage_label="oncology_specialist",
            rejection_threshold=ONCOLOGY_REJECTION_THRESHOLD,
            margin_thresholds=margin_thresholds,
        )

    def predict(self, title: str, top_k: int = DEFAULT_TOP_K) -> List[PredictionResult]:
        return self._inner.predict(title, top_k)


# ──────────────────────────────────────────────────────────────
# HIERARCHICAL DOMAIN INFERENCE  (orchestrateur principal)
# ──────────────────────────────────────────────────────────────

class HierarchicalDomainInference:
    """
    Moteur d'inférence hiérarchique à deux niveaux.

    Niveau 1 — Routage général :
        Un centroïde méta-oncologie (barycentre de tous les centroïdes
        de la famille oncologique) permet de calculer un score d'appartenance
        globale à l'oncologie AVANT la classification fine.

    Niveau 2 — Spécialisation conditionnelle :
        Si meta_score ≥ oncology_routing_threshold
          → OncologySpecialistClassifier (centroïdes oncologie uniquement)
        Sinon
          → DomainClassifier général (tous les domaines)

    Avantages :
    ✔ Hors oncologie  : le seuil de rejet (REJECTION_THRESHOLD) filtre les
                        titres sans rapport avec les domaines connus.
    ✔ En oncologie    : le spécialiste discrimine finement recherche
                        vs clinique, et sous-spécialités.
    ✔ Frontière       : la marge et le meta_score permettent de détecter
                        les cas ambigus (ex. épidémiologie du cancer).
    """

    def __init__(
        self,
        embed_model,
        oncology_family_key: str = "oncology",
        oncology_routing_threshold: float = ONCOLOGY_ROUTING_THRESHOLD,
        positional_decay: float = POSITIONAL_DECAY,
        temperature: float = CONFIDENCE_TEMPERATURE,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialise le moteur d'inférence hiérarchique.

        La taxonomie est embarquée dans le module via DOMAIN_CATALOG —
        aucune dépendance externe (plus de taxonomy.py).

        Args:
            oncology_family_key       : Valeur de DomainFamily.value pour l'oncologie.
            oncology_routing_threshold: Score méta au-delà duquel le spécialiste
                                        oncologie est activé.
            positional_decay          : Taux de décroissance pondérale positionnelle.
            temperature               : Température du softmax de calibration.
            cache_dir                 : Dossier où persister les centroïdes (.npy).
                                        None → même dossier que domain_inference.py.
        """
        self._oncology_family_key = oncology_family_key
        self._routing_threshold = oncology_routing_threshold
        self._positional_decay = positional_decay

        self._model = embed_model
        self._backend_type: str = backend.get_embed_backend()
        self._scorer = SimilarityScorer(temperature=temperature)
        self._builder = CentroidBuilder(self._model, positional_decay=positional_decay)

        # Partition du catalogue embarqué : oncologie vs reste
        self._all_domains: Dict[str, ScienceDomain] = DOMAIN_CATALOG
        self._oncology_domains: Dict[str, ScienceDomain] = {
            d_id: d
            for d_id, d in self._all_domains.items()
            if d.family.value == oncology_family_key
        }

        # Centroïdes et classifieurs
        self._general_centroids: Dict[str, np.ndarray] = {}
        self._oncology_centroids: Dict[str, np.ndarray] = {}
        self._oncology_meta_centroid: Optional[np.ndarray] = None
        self._general_classifier: Optional[DomainClassifier] = None
        self._oncology_specialist: Optional[OncologySpecialistClassifier] = None
        # Seuils de marge adaptatifs (calculés après construction des centroïdes)
        self._general_margin_thresholds: Dict[str, float] = {}
        self._oncology_margin_thresholds: Dict[str, float] = {}

        # Répertoire de cache
        self._cache_dir: Path = (
            Path(cache_dir) if cache_dir else Path(__file__).parent
        )

        if self._model is not None:
            self._initialize()
        else:
            print("⚠️ [HierarchicalDomainInference] Modèle non chargé. Inférence désactivée.")

    # ── Initialisation ───────────────────────────────────────

    def _taxonomy_fingerprint(self) -> str:
        """
        Produit un hash SHA-256 (8 caractères) représentant l'état du catalogue
        et des hyperparamètres qui influencent les centroïdes.

        Composantes du hash :
          - Toutes les descriptions de chaque domaine de DOMAIN_CATALOG (ordre préservé)
          - POSITIONAL_DECAY (change la pondération des embeddings)

        Si tu ajoutes un domaine dans DOMAIN_CATALOG ou modifies une description,
        le fingerprint change → cache invalidé automatiquement au prochain démarrage.
        """
        h = hashlib.sha256()
        for d_id in sorted(self._all_domains.keys()):
            domain = self._all_domains[d_id]
            for desc in domain.descriptions:
                h.update(desc.encode("utf-8"))
        h.update(str(self._positional_decay).encode())
        return h.hexdigest()[:8]

    def _cache_paths(self, fingerprint: str) -> tuple[Path, Path, Path]:
        """Retourne les chemins des 3 fichiers .npy du cache."""
        base = self._cache_dir / f".centroids_{fingerprint}"
        return (
            Path(str(base) + "_general.npy"),
            Path(str(base) + "_oncology.npy"),
            Path(str(base) + "_meta.npy"),
        )

    def _try_load_cache(
        self, fingerprint: str
    ) -> Optional[tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]]:
        """
        Tente de charger les centroïdes depuis le cache disque.

        Vérifie que le nombre d'entrées correspond à la taxonomie actuelle
        pour éviter un cache partiellement corrompu.

        Returns:
            (general_centroids, oncology_centroids, meta_centroid) ou None si échec.
        """
        p_gen, p_onc, p_meta = self._cache_paths(fingerprint)
        if not (p_gen.exists() and p_onc.exists() and p_meta.exists()):
            return None
        try:
            gen_raw = np.load(p_gen, allow_pickle=True).item()
            onc_raw = np.load(p_onc, allow_pickle=True).item()
            meta = np.load(p_meta)

            # Sanity check : le nombre de domaines doit correspondre
            if len(gen_raw) != len(self._all_domains):
                print(
                    f"⚠️ [Cache] Taille incohérente general ({len(gen_raw)} vs "
                    f"{len(self._all_domains)}) — recalcul."
                )
                return None
            if len(onc_raw) != len(self._oncology_domains):
                print(
                    f"⚠️ [Cache] Taille incohérente oncology ({len(onc_raw)} vs "
                    f"{len(self._oncology_domains)}) — recalcul."
                )
                return None

            # Conversion explicite float32 après chargement
            gen: Dict[str, np.ndarray] = {k: v.astype(np.float32) for k, v in gen_raw.items()}
            onc: Dict[str, np.ndarray] = {k: v.astype(np.float32) for k, v in onc_raw.items()}
            return gen, onc, meta.astype(np.float32)
        except Exception as exc:
            print(f"⚠️ [Cache] Lecture impossible ({exc}) — recalcul.")
            return None

    def _save_cache(
        self,
        fingerprint: str,
        gen: Dict[str, np.ndarray],
        onc: Dict[str, np.ndarray],
        meta: np.ndarray,
    ) -> None:
        """
        Persiste les centroïdes sur disque.
        En cas d'erreur (droits, espace), logue un avertissement sans planter.
        """
        p_gen, p_onc, p_meta = self._cache_paths(fingerprint)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(p_gen, gen)
            np.save(p_onc, onc)
            np.save(p_meta, meta)
            print(f"   💾 Cache centroïdes sauvegardé → {p_gen.parent} (clé: {fingerprint})")
        except Exception as exc:
            print(f"   ⚠️ [Cache] Impossible de sauvegarder ({exc}) — cache ignoré.")

    def _initialize(self) -> None:
        """
        Construit (ou restaure depuis le cache disque) les centroïdes et
        instancie les classifieurs avec leurs seuils de marge adaptatifs.

        Flux :
          1. Calcul du fingerprint de la taxonomie (hash SHA-256 tronqué).
          2. Tentative de chargement depuis le cache .npy correspondant.
          3. Si succès → log "Cache hit" et skip de l'encodage E5.
          4. Si échec → encodage complet + sauvegarde du cache pour la prochaine fois.
          5. Dans tous les cas → calcul des seuils de marge adaptatifs (rapide,
             uniquement produits scalaires entre centroïdes) puis instanciation
             des classifieurs avec ces seuils.
        """
        fingerprint = self._taxonomy_fingerprint()

        cached = self._try_load_cache(fingerprint)
        if cached is not None:
            self._general_centroids, self._oncology_centroids, self._oncology_meta_centroid = cached
            print(
                f"🚀 [HierarchicalDomainInference] Cache hit (clé {fingerprint}) — "
                f"{len(self._general_centroids)} centroïdes généraux, "
                f"{len(self._oncology_centroids)} oncologiques."
            )
        else:
            print(
                f"🔧 [HierarchicalDomainInference] Cache miss (clé {fingerprint}) — "
                "Construction des centroïdes par encodage E5…"
            )
            self._general_centroids = self._builder.build(self._all_domains)
            self._oncology_centroids = self._builder.build(self._oncology_domains)
            self._oncology_meta_centroid = self._build_meta_centroid(self._oncology_domains)
            self._save_cache(
                fingerprint,
                self._general_centroids,
                self._oncology_centroids,
                self._oncology_meta_centroid,
            )
            print(f"   ✅ {len(self._general_centroids)} centroïdes généraux")
            print(f"   ✅ {len(self._oncology_centroids)} centroïdes oncologiques")
            print(f"   ✅ Centroïde méta-oncologie calculé")

        # Seuils adaptatifs : O(n²) produits scalaires, négligeable vs encodage E5
        self._general_margin_thresholds = self._compute_adaptive_margin_thresholds(
            self._general_centroids
        )
        self._oncology_margin_thresholds = self._compute_adaptive_margin_thresholds(
            self._oncology_centroids
        )
        print(
            f"   ✅ Seuils de marge adaptatifs calculés "
            f"(général: {len(self._general_margin_thresholds)}, "
            f"oncologie: {len(self._oncology_margin_thresholds)})"
        )

        self._general_classifier = DomainClassifier(
            self._all_domains, self._general_centroids, self._scorer,
            margin_thresholds=self._general_margin_thresholds,
        )
        self._oncology_specialist = OncologySpecialistClassifier(
            self._oncology_domains, self._oncology_centroids, self._scorer,
            margin_thresholds=self._oncology_margin_thresholds,
        )

    def _compute_adaptive_margin_thresholds(
        self,
        centroids: Dict[str, np.ndarray],
        alpha: float = 0.30,
    ) -> Dict[str, float]:
        """
        Calcule un seuil de marge adaptatif pour chaque domaine, proportionnel
        à la distance cosinus minimale vers son plus proche voisin centroïde.

        Principe (Piste 3) :
          Si deux centroïdes sont naturellement proches (ex. anapath / onco_anapath),
          une faible marge entre eux est normale et ne doit pas déclencher un rejet.
          Si un centroïde est isolé (ex. géomatique dans le catalogue CLB),
          une faible marge est suspecte → seuil élevé.

        Formule :
          seuil_i = clip(alpha × min_{j≠i}(1 − dot(c_i, c_j)),
                         MARGIN_REJECTION_THRESHOLD / 2,   # plancher 0.02
                         MARGIN_REJECTION_THRESHOLD * 2)   # plafond  0.08

        alpha = 0.30 : empiriquement, 30% de la distance au plus proche voisin
        représente une ambiguïté structurelle non résoluble par E5.

        Complexité : O(n²) en produits scalaires — négligeable (~1 ms pour 40 domaines).

        Args:
            centroids : Dictionnaire {domain_id → vecteur L2-normalisé}.
            alpha     : Fraction de la distance min utilisée comme seuil.

        Returns:
            Dictionnaire {domain_id → seuil_marge_adaptatif}.
        """
        ids: List[str] = list(centroids.keys())
        thresholds: Dict[str, float] = {}

        for d_id in ids:
            c_i: np.ndarray = centroids[d_id]
            min_dist: float = float("inf")

            for other_id in ids:
                if other_id == d_id:
                    continue
                # distance cosinus = 1 − similarité cosinus (centroïdes L2-normalisés)
                dist: float = 1.0 - float(np.dot(c_i, centroids[other_id]))
                if dist < min_dist:
                    min_dist = dist

            raw: float = alpha * min_dist if min_dist < float("inf") else MARGIN_REJECTION_THRESHOLD
            thresholds[d_id] = float(np.clip(
                raw,
                MARGIN_REJECTION_THRESHOLD / 2,   # plancher : 0.02
                MARGIN_REJECTION_THRESHOLD * 2,   # plafond  : 0.08
            ))

        return thresholds

    def _build_meta_centroid(self, domains: Dict[str, ScienceDomain]) -> np.ndarray:
        """
        Centroïde méta = barycentre (non pondéré) des centroïdes du groupe.
        Représente le "centre de gravité sémantique" de toute la famille.

        Implémentation : on moyenne les centroïdes déjà calculés dans
        self._general_centroids (même espace vectoriel), puis on re-normalise.
        """
        vecs: List[np.ndarray] = [
            self._general_centroids[d_id]
            for d_id in domains
            if d_id in self._general_centroids
        ]
        if not vecs:
            return np.zeros(1)

        meta: np.ndarray = np.mean(np.stack(vecs), axis=0)
        norm: float = float(np.linalg.norm(meta))
        return meta / norm if norm > 0 else meta

    # ── Interface publique ────────────────────────────────────

    def predict(self, title: str, top_k: int = DEFAULT_TOP_K) -> ClassificationReport:
        """
        Point d'entrée principal. Orchestre le routage hiérarchique.

        Returns:
            ClassificationReport avec prédictions enrichies et métadonnées.
        """
        if self._general_classifier is None:
            return ClassificationReport(
                query=title, predictions=[],
                oncology_meta_score=0.0, routed_to_specialist=False,
            )

        # 1. Score d'appartenance globale à l'oncologie
        meta_score: float = self._general_classifier.score_against_meta(
            title, self._oncology_meta_centroid
        )

        # 2. Routage conditionnel
        use_specialist: bool = (
            meta_score >= self._routing_threshold
            and self._oncology_specialist is not None
            and bool(self._oncology_domains)
        )

        predictions: List[PredictionResult] = (
            self._oncology_specialist.predict(title, top_k)
            if use_specialist
            else self._general_classifier.predict(title, top_k)
        )

        return ClassificationReport(
            query=title,
            predictions=predictions,
            oncology_meta_score=meta_score,
            routed_to_specialist=use_specialist,
        )


# ──────────────────────────────────────────────────────────────
# Alias de compatibilité ascendante
# Permet de remplacer DomainInference v1.0 sans modifier le reste du projet.
# ──────────────────────────────────────────────────────────────
DomainInference = HierarchicalDomainInference


# ──────────────────────────────────────────────────────────────
# TEST UNITAIRE & VALIDATION
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m, b_type = backend.load_embed_model()
    backend.set_embed_model(m, b_type)

    engine = HierarchicalDomainInference()

    test_titles: List[Tuple[str, str]] = [
        # (titre, catégorie attendue)
        ("Efficacy of Pembrolizumab in Non-Small Cell Lung Cancer: A Phase III Trial",
         "oncologie clinique"),
        ("Deep Learning for Automated Segmentation of Organs at Risk in Head and Neck Radiotherapy",
         "oncologie recherche (IA)"),
        ("Impact of Pesticide Exposure on Cancer Risk in Rural Populations: A Systematic Review",
         "épidémiologie / frontière"),
        ("A New Mathematical Approach for Modeling Tumor Growth Dynamics",
         "mathématiques / frontière"),
        ("Quantum Entanglement in Superconducting Qubits at Ultra-Low Temperatures",
         "hors domaine attendu"),
    ]

    print(f"\n🚀 Test — HierarchicalDomainInference (Backend: {b_type.upper()})")
    print("=" * 72)

    for title, expected_category in test_titles:
        report: ClassificationReport = engine.predict(title, top_k=2)

        route_tag: str = "🎯 SPÉCIALISTE ONCOLOGIE" if report.routed_to_specialist else "🌐 GÉNÉRAL"
        meta_pct: str = f"{report.oncology_meta_score * 100:.1f}%"

        print(f"\n📄 {title}")
        print(f"   Catégorie attendue   : {expected_category}")
        print(f"   Méta-oncologie       : {meta_pct}  |  Routage : {route_tag}")

        if report.is_out_of_scope:
            print("   ⚠️  HORS DOMAINE (score < seuil de rejet)")
        else:
            for pred in report.predictions:
                conf_str: str = f"{pred.confidence * 100:.1f}%"
                margin_str: str = f"Δ={pred.margin:.3f}"
                oos_flag: str = " ⚠️ OOS" if pred.is_out_of_scope else ""
                print(
                    f"   └─ [{conf_str}] {pred.domain.family.value} > {pred.domain.id}"
                    f"  ({margin_str}){oos_flag}"
                )