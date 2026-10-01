{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4eZ42kpoJ_RY",
   "metadata": {
    "id": "4eZ42kpoJ_RY"
   },
   "source": [
    "# Bioactive Molecule Prediction Using Extreme Gradient Boosting — V2 Pipeline\n",
    "\n",
    "**Target:** ChEMBL230 bioactivity dataset (IC50 assay records)\n",
    "\n",
    "**Base paper:** Mustapha & Saeed, *Molecules* 2016\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "V5nZjo-TJ_Rc",
   "metadata": {
    "id": "V5nZjo-TJ_Rc"
   },
   "source": [
    "## Section 2 — Imports and Configuration\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "HhDMX4OkJ_Rd",
   "metadata": {
    "id": "HhDMX4OkJ_Rd"
   },
   "outputs": [],
   "source": [
    "\n",
    "!pip install -q rdkit xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "_7OifMKrJ_Re",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "_7OifMKrJ_Re",
    "outputId": "096029d7-4680-41a3-90a2-367eca768e4e"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Libraries imported successfully.\n",
      "Random state       : 42\n",
      "Fingerprint radius : 2\n",
      "Fingerprint bits   : 2048\n",
      "Holdout fraction   : 0.3\n",
      "CV folds           : 10\n"
     ]
    }
   ],
   "source": [
    "# ----- Core data handling -----\n",
    "import json\n",
    "import time\n",
    "import glob\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# ----- Chemistry -----\n",
    "from rdkit import Chem\n",
    "from rdkit.Chem import AllChem\n",
    "from rdkit import RDLogger\n",
    "RDLogger.DisableLog(\"rdApp.*\")   # silence RDKit's noisy SMILES-parsing warnings\n",
    "\n",
    "# ----- Modelling -----\n",
    "from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.calibration import CalibratedClassifierCV\n",
    "from sklearn.naive_bayes import BernoulliNB\n",
    "from sklearn.kernel_approximation import RBFSampler\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score,\n",
    "    f1_score, roc_auc_score, confusion_matrix\n",
    ")\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "import joblib\n",
    "\n",
    "# ----- Project configuration (single source of truth) -----\n",
    "RANDOM_STATE = 42\n",
    "FINGERPRINT_RADIUS = 2\n",
    "FINGERPRINT_BITS = 2048\n",
    "HOLDOUT_FRACTION = 0.30\n",
    "N_CV_FOLDS = 10\n",
    "\n",
    "np.random.seed(RANDOM_STATE)\n",
    "\n",
    "print(\"Libraries imported successfully.\")\n",
    "print(f\"Random state       : {RANDOM_STATE}\")\n",
    "print(f\"Fingerprint radius : {FINGERPRINT_RADIUS}\")\n",
    "print(f\"Fingerprint bits   : {FINGERPRINT_BITS}\")\n",
    "print(f\"Holdout fraction   : {HOLDOUT_FRACTION}\")\n",
    "print(f\"CV folds           : {N_CV_FOLDS}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7W6ATQ1rJ_Rf",
   "metadata": {
    "id": "7W6ATQ1rJ_Rf"
   },
   "source": [
    "## Section 3 — Load the Raw Dataset Only\n",
    "\n",
    "**What:** Load `CHEMBL230_RAW_DATA.csv`, the raw ChEMBL export.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "egUqoivBJ_Rf",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "egUqoivBJ_Rf",
    "outputId": "d97002b6-a912-45e1-c416-9e562f1cd8d2"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Raw file: CHEMBL230_RAW_DATA.csv\n",
      "\n",
      "Raw dataset shape: (7979, 5)\n"
     ]
    }
   ],
   "source": [
    "def find_file(pattern_list):\n",
    "    \"\"\"Return the first existing file that matches any of the given glob patterns.\"\"\"\n",
    "    for pattern in pattern_list:\n",
    "        matches = sorted(glob.glob(pattern))\n",
    "        if matches:\n",
    "            return matches[0]\n",
    "    raise FileNotFoundError(f\"No file found for patterns: {pattern_list}\")\n",
    "\n",
    "raw_path = find_file([\"CHEMBL230_RAW_DATA*.csv\", \"*RAW*DATA*.csv\"])\n",
    "\n",
    "print(\"Raw file:\", raw_path)\n",
    "\n",
    "raw_df = pd.read_csv(raw_path)\n",
    "\n",
    "print()\n",
    "print(\"Raw dataset shape:\", raw_df.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2yq1vHOHJ_Rg",
   "metadata": {
    "id": "2yq1vHOHJ_Rg"
   },
   "source": [
    "## Section 4 — Inspect the Raw Data\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "VDv7mLFCJ_Rg",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VDv7mLFCJ_Rg",
    "outputId": "008ea49c-e085-4145-bd54-73ac440ad9e0"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Columns:\n",
      " - Molecule ChEMBL ID\n",
      " - Smiles\n",
      " - Standard Type\n",
      " - Standard Value\n",
      " - Standard Units\n",
      "\n",
      "Data types:\n",
      "Molecule ChEMBL ID     object\n",
      "Smiles                 object\n",
      "Standard Type          object\n",
      "Standard Value        float64\n",
      "Standard Units         object\n",
      "dtype: object\n",
      "\n",
      "Missing values per column:\n",
      "Molecule ChEMBL ID       0\n",
      "Smiles                 190\n",
      "Standard Type          157\n",
      "Standard Value        1139\n",
      "Standard Units        1120\n",
      "dtype: int64\n",
      "\n",
      "Fully duplicated rows: 602\n",
      "\n",
      "Standard Type value counts:\n",
      "Standard Type\n",
      "IC50    7822\n",
      "NaN      157\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Standard Units value counts:\n",
      "Standard Units\n",
      "nM         6832\n",
      "NaN        1120\n",
      "ug.mL-1      25\n",
      "ug            1\n",
      "%             1\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Standard Value summary statistics:\n",
      "count    6.840000e+03\n",
      "mean     1.063770e+06\n",
      "std      1.977218e+07\n",
      "min      0.000000e+00\n",
      "25%      1.000000e+02\n",
      "50%      9.705000e+02\n",
      "75%      1.000000e+04\n",
      "max      8.333357e+08\n",
      "Name: Standard Value, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(\"Columns:\")\n",
    "for c in raw_df.columns:\n",
    "    print(\" -\", c)\n",
    "\n",
    "print()\n",
    "print(\"Data types:\")\n",
    "print(raw_df.dtypes)\n",
    "\n",
    "print()\n",
    "print(\"Missing values per column:\")\n",
    "print(raw_df.isna().sum())\n",
    "\n",
    "print()\n",
    "print(\"Fully duplicated rows:\", raw_df.duplicated().sum())\n",
    "\n",
    "print()\n",
    "print(\"Standard Type value counts:\")\n",
    "print(raw_df[\"Standard Type\"].value_counts(dropna=False))\n",
    "\n",
    "print()\n",
    "print(\"Standard Units value counts:\")\n",
    "print(raw_df[\"Standard Units\"].value_counts(dropna=False))\n",
    "\n",
    "print()\n",
    "print(\"Standard Value summary statistics:\")\n",
    "print(raw_df[\"Standard Value\"].describe())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PUn74eJOJ_Rh",
   "metadata": {
    "id": "PUn74eJOJ_Rh"
   },
   "source": [
    "## Section 5 — Raw Data Preprocessing\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "vt4kkZMNJ_Rh",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vt4kkZMNJ_Rh",
    "outputId": "c50389da-b21f-44bf-e1b1-3129a1ee6367"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Rows before cleaning                       : 7979\n",
      "Rows after removing missing Value / Smiles  : 6839\n",
      "Rows removed                                : 1140\n"
     ]
    }
   ],
   "source": [
    "rows_before = len(raw_df)\n",
    "\n",
    "clean_df = raw_df.dropna(subset=[\"Standard Value\", \"Smiles\"]).reset_index(drop=True)\n",
    "\n",
    "rows_after = len(clean_df)\n",
    "\n",
    "print(f\"Rows before cleaning                       : {rows_before}\")\n",
    "print(f\"Rows after removing missing Value / Smiles  : {rows_after}\")\n",
    "print(f\"Rows removed                                : {rows_before - rows_after}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "rlEIqFBIJ_Ri",
   "metadata": {
    "id": "rlEIqFBIJ_Ri"
   },
   "source": [
    "## Section 6 — SMILES Validation with RDKit\n",
    "\n",
    "**What:** Check that every remaining SMILES string can actually be\n",
    "parsed into a chemical structure by RDKit.\n",
    "\n",
    "**Why:** A Morgan fingerprint can only be generated from a SMILES\n",
    "string that RDKit can successfully turn into a molecule object. A\n",
    "string that fails to parse cannot be used later, so we identify (and\n",
    "only then remove) such rows — nothing is dropped silently.\n",
    "\n",
    "**Output:** Count of valid vs. invalid SMILES, and the final row count\n",
    "after removing any invalid ones (only if invalid ones actually exist).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "Rz5WNUekJ_Ri",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Rz5WNUekJ_Ri",
    "outputId": "4b293d17-30a9-4e7f-e17d-230d7c8e02bd"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Valid SMILES   : 6839\n",
      "Invalid SMILES : 0\n",
      "No invalid SMILES found — nothing removed at this step.\n"
     ]
    }
   ],
   "source": [
    "def is_valid_smiles(smiles):\n",
    "    \"\"\"Return True if RDKit can parse this SMILES string into a molecule.\"\"\"\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    return mol is not None\n",
    "\n",
    "clean_df[\"smiles_is_valid\"] = clean_df[\"Smiles\"].apply(is_valid_smiles)\n",
    "\n",
    "n_valid = clean_df[\"smiles_is_valid\"].sum()\n",
    "n_invalid = (~clean_df[\"smiles_is_valid\"]).sum()\n",
    "\n",
    "print(f\"Valid SMILES   : {n_valid}\")\n",
    "print(f\"Invalid SMILES : {n_invalid}\")\n",
    "\n",
    "if n_invalid > 0:\n",
    "    print()\n",
    "    print(\"Removing invalid SMILES rows...\")\n",
    "    clean_df = clean_df[clean_df[\"smiles_is_valid\"]].reset_index(drop=True)\n",
    "    print(f\"Rows remaining after removing invalid SMILES: {len(clean_df)}\")\n",
    "else:\n",
    "    print(\"No invalid SMILES found — nothing removed at this step.\")\n",
    "\n",
    "clean_df = clean_df.drop(columns=[\"smiles_is_valid\"])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aDYZo3U0J_Ri",
   "metadata": {
    "id": "aDYZo3U0J_Ri"
   },
   "source": [
    "## Section 7 — Create the Activity Label from Standard Value\n",
    "\n",
    "**Activity rule:**\n",
    "- `Standard Value < 10,000` → `Active`\n",
    "- `Standard Value >= 10,000` → `Inactive`\n",
    "\n",
    "Therefore, an exact value of `10,000` is classified as `Inactive`.\n",
    "\n",
    "**Important:** `Standard Value` is used here only to create the target label. It must **not** enter the feature matrix `X` later, because that would cause target leakage.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "Kol5paVEJ_Ri",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Kol5paVEJ_Ri",
    "outputId": "0881cecf-143c-403e-ae35-e85fa554b361"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Activity rule:\n",
      "  Standard Value < 10,000  -> Active\n",
      "  Standard Value >= 10,000 -> Inactive\n",
      "\n",
      "Records with Standard Value exactly 10,000 : 228\n",
      "Active records                           : 5004\n",
      "Inactive records                         : 1835\n",
      "\n",
      "Activity distribution:\n",
      "Activity\n",
      "Active      5004\n",
      "Inactive    1835\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Activity labels were created directly from the raw Standard Value.\n",
      "No V1/reference dataset was used.\n"
     ]
    }
   ],
   "source": [
    "# Create the Activity target directly from the raw Standard Value.\n",
    "# This is the only place where Standard Value determines the target label.\n",
    "\n",
    "clean_df[\"Activity\"] = np.where(\n",
    "    clean_df[\"Standard Value\"] < 10000,\n",
    "    \"Active\",\n",
    "    \"Inactive\",\n",
    ")\n",
    "\n",
    "n_exact_10000 = int((clean_df[\"Standard Value\"] == 10000).sum())\n",
    "n_active = int((clean_df[\"Activity\"] == \"Active\").sum())\n",
    "n_inactive = int((clean_df[\"Activity\"] == \"Inactive\").sum())\n",
    "\n",
    "print(\"Activity rule:\")\n",
    "print(\"  Standard Value < 10,000  -> Active\")\n",
    "print(\"  Standard Value >= 10,000 -> Inactive\")\n",
    "print()\n",
    "print(f\"Records with Standard Value exactly 10,000 : {n_exact_10000}\")\n",
    "print(f\"Active records                           : {n_active}\")\n",
    "print(f\"Inactive records                         : {n_inactive}\")\n",
    "print()\n",
    "print(\"Activity distribution:\")\n",
    "print(clean_df[\"Activity\"].value_counts())\n",
    "\n",
    "if n_active + n_inactive != len(clean_df):\n",
    "    raise ValueError(\"Activity labeling did not cover every cleaned record.\")\n",
    "\n",
    "print()\n",
    "print(\"Activity labels were created directly from the raw Standard Value.\")\n",
    "print(\"No V1/reference dataset was used.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "JAlND2QXJ_Rj",
   "metadata": {
    "id": "JAlND2QXJ_Rj"
   },
   "source": [
    "## Section 8 — Assay Records vs. Molecule IDs\n",
    "\n",
    "**What:** Confirm we still have one row per **assay record**, not one\n",
    "row per molecule, and look at how many molecules have more than one\n",
    "record (and how many even have conflicting Active/Inactive labels\n",
    "across their different records).\n",
    "\n",
    "**Why:** A molecule tested in several assays can legitimately show\n",
    "different results. Collapsing those into a single row per molecule\n",
    "(e.g. with `groupby(...).median()` or `drop_duplicates` on the\n",
    "molecule ID) would throw away real experimental information. We keep\n",
    "every assay record; the molecule ID is used later **only** to keep a\n",
    "molecule's records together during splitting, never to merge rows.\n",
    "\n",
    "**Output:** Total records, total unique molecules, how many molecules\n",
    "have multiple records, and how many of those have both an Active and\n",
    "an Inactive record.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "AmhNtkOXJ_Rj",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "AmhNtkOXJ_Rj",
    "outputId": "7194b1d8-c556-4eeb-af42-85a9ef2d48f7"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Total assay records                                 : 6839\n",
      "Total unique molecules                              : 5030\n",
      "Molecules with more than one assay record           : 956\n",
      "Molecules with both Active and Inactive records     : 151\n",
      "\n",
      "All of the above records are kept as-is. Nothing is collapsed or\n",
      "deduplicated by molecule ID.\n"
     ]
    }
   ],
   "source": [
    "from rdkit.Chem.rdmolops import Cleanup\n",
    "n_records = len(clean_df)\n",
    "n_molecules = clean_df[\"Molecule ChEMBL ID\"].nunique()\n",
    "\n",
    "records_per_molecule = clean_df.groupby(\"Molecule ChEMBL ID\").size()\n",
    "n_molecules_multi_record = int((records_per_molecule > 1).sum())\n",
    "\n",
    "labels_per_molecule = clean_df.groupby(\"Molecule ChEMBL ID\")[\"Activity\"].nunique()\n",
    "n_molecules_conflicting = int((labels_per_molecule > 1).sum())\n",
    "\n",
    "print(f\"Total assay records                                 : {n_records}\")\n",
    "print(f\"Total unique molecules                              : {n_molecules}\")\n",
    "print(f\"Molecules with more than one assay record           : {n_molecules_multi_record}\")\n",
    "print(f\"Molecules with both Active and Inactive records     : {n_molecules_conflicting}\")\n",
    "print()\n",
    "print(\"All of the above records are kept as-is. Nothing is collapsed or\")\n",
    "print(\"deduplicated by molecule ID.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "mOoZQhKGJ_Rj",
   "metadata": {
    "id": "mOoZQhKGJ_Rj"
   },
   "source": [
    "## Section 9 — Save the Final V2 Preprocessed Dataset\n",
    "\n",
    "**What:** Keep the useful columns and write the cleaned, activity-\n",
    "labelled dataset to `CHEMBL230_Preprocessed_Data_V2.csv`.\n",
    "\n",
    "**Why:** This file is our checkpoint — everything from here on\n",
    "(fingerprints, splitting, modelling) is built from this saved file, so\n",
    "anyone re-running the notebook can start from this point directly.\n",
    "\n",
    "**Output:** Final shape and Activity distribution, plus the saved CSV file.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "VWoHxfJ0J_Rj",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VWoHxfJ0J_Rj",
    "outputId": "b37aa08d-0b4b-4113-84e6-b202a9d85d3d"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Saved CHEMBL230_Preprocessed_Data_V2.csv\n",
      "Final shape: (6839, 6)\n",
      "\n",
      "Activity distribution:\n",
      "Activity\n",
      "Active      5004\n",
      "Inactive    1835\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "output_columns = [\n",
    "    \"Molecule ChEMBL ID\", \"Smiles\", \"Standard Type\",\n",
    "    \"Standard Value\", \"Standard Units\", \"Activity\",\n",
    "]\n",
    "\n",
    "v2_df = clean_df[output_columns].reset_index(drop=True)\n",
    "\n",
    "v2_df.to_csv(\"CHEMBL230_Preprocessed_Data_V2.csv\", index=False)\n",
    "\n",
    "print(\"Saved CHEMBL230_Preprocessed_Data_V2.csv\")\n",
    "print(\"Final shape:\", v2_df.shape)\n",
    "print()\n",
    "print(\"Activity distribution:\")\n",
    "print(v2_df[\"Activity\"].value_counts())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "KrVeMJNoJ_Rj",
   "metadata": {
    "id": "KrVeMJNoJ_Rj"
   },
   "source": [
    "## Section 10 — SMILES → Morgan Fingerprint\n",
    "\n",
    "**What:** Convert every SMILES string into a 2048-bit Morgan\n",
    "fingerprint (an ECFP4-style circular fingerprint), which is the\n",
    "numeric representation the six ML models will actually be trained on.\n",
    "\n",
    "**Why:** Classical ML models like XGBoost, Random Forest, SVM, etc.\n",
    "need fixed-length numeric input — they cannot read a SMILES string\n",
    "directly. A Morgan fingerprint encodes, as a fixed-length bit vector,\n",
    "which small circular substructures (of the given radius) are present\n",
    "in the molecule:\n",
    "\n",
    "```\n",
    "SMILES  ->  RDKit molecule  ->  Morgan fingerprint  ->  2048-bit vector  ->  ML model\n",
    "```\n",
    "\n",
    "**How:** `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)`\n",
    "for every row, then stack the bit vectors into one big matrix `X`.\n",
    "`y` is the Activity label encoded as `Active = 1`, `Inactive = 0`.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "pZHi6w1cJ_Rj",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "pZHi6w1cJ_Rj",
    "outputId": "a07e5f14-6fc0-4256-9b0c-ba21e5338430"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Morgan fingerprints for all 6839 records...\n",
      "Done in 5.9 seconds.\n",
      "\n",
      "X shape: (6839, 2048)\n",
      "y shape: (6839,)\n",
      "\n",
      "Class distribution (1 = Active, 0 = Inactive):\n",
      "1    5004\n",
      "0    1835\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "def smiles_to_fingerprint(smiles, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_BITS):\n",
    "    \"\"\"Convert one SMILES string into a Morgan fingerprint as a numpy array of 0/1.\"\"\"\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)\n",
    "    array = np.zeros((n_bits,), dtype=np.int8)\n",
    "    Chem.DataStructs.ConvertToNumpyArray(fingerprint, array)\n",
    "    return array\n",
    "\n",
    "print(\"Generating Morgan fingerprints for all\", len(v2_df), \"records...\")\n",
    "start_time = time.time()\n",
    "\n",
    "fingerprint_list = [smiles_to_fingerprint(s) for s in v2_df[\"Smiles\"]]\n",
    "X = np.vstack(fingerprint_list)\n",
    "\n",
    "elapsed = time.time() - start_time\n",
    "print(f\"Done in {elapsed:.1f} seconds.\")\n",
    "\n",
    "y = (v2_df[\"Activity\"] == \"Active\").astype(int).values\n",
    "molecule_ids = v2_df[\"Molecule ChEMBL ID\"].values\n",
    "\n",
    "print()\n",
    "print(\"X shape:\", X.shape)\n",
    "print(\"y shape:\", y.shape)\n",
    "print()\n",
    "print(\"Class distribution (1 = Active, 0 = Inactive):\")\n",
    "print(pd.Series(y).value_counts())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "RS4I6ASZJ_Rk",
   "metadata": {
    "id": "RS4I6ASZJ_Rk"
   },
   "source": [
    "## Section 11 — Molecule-Aware 70/30 Development / Holdout Split\n",
    "\n",
    "**What:** Split the data into a 70% development set (used for model\n",
    "comparison and selection) and a 30% holdout set (touched only once,\n",
    "at the very end, for the final evaluation).\n",
    "\n",
    "**Why:** Because the same molecule can have several assay records, a\n",
    "plain random split could put some records of a molecule in\n",
    "development and other records of the *same* molecule in holdout. That\n",
    "would let information about a molecule leak between the two sets. We\n",
    "use `GroupShuffleSplit` with `Molecule ChEMBL ID` as the group, which\n",
    "guarantees every record of a given molecule stays on the same side of\n",
    "the split.\n",
    "\n",
    "**Output:** Record and molecule counts on each side, the actual\n",
    "percentage split achieved, and — most importantly — the molecule\n",
    "overlap count, which must be zero.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "KLvWTuvgJ_Rk",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "KLvWTuvgJ_Rk",
    "outputId": "49742c9c-ab71-40db-da87-fc73e78637f0"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Development records  : 4830\n",
      "Holdout records       : 2009\n",
      "Development molecules : 3521\n",
      "Holdout molecules     : 1509\n",
      "Holdout percentage    : 29.4%\n",
      "Molecule overlap       : 0\n",
      "\n",
      "Molecule overlap = 0  --  confirmed clean split.\n"
     ]
    }
   ],
   "source": [
    "splitter = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)\n",
    "dev_idx, holdout_idx = next(splitter.split(X, y, groups=molecule_ids))\n",
    "\n",
    "X_dev, X_holdout = X[dev_idx], X[holdout_idx]\n",
    "y_dev, y_holdout = y[dev_idx], y[holdout_idx]\n",
    "groups_dev = molecule_ids[dev_idx]\n",
    "groups_holdout = molecule_ids[holdout_idx]\n",
    "\n",
    "dev_molecules = set(groups_dev)\n",
    "holdout_molecules = set(groups_holdout)\n",
    "overlap = dev_molecules & holdout_molecules\n",
    "\n",
    "print(f\"Development records  : {len(dev_idx)}\")\n",
    "print(f\"Holdout records       : {len(holdout_idx)}\")\n",
    "print(f\"Development molecules : {len(dev_molecules)}\")\n",
    "print(f\"Holdout molecules     : {len(holdout_molecules)}\")\n",
    "print(f\"Holdout percentage    : {len(holdout_idx) / len(X):.1%}\")\n",
    "print(f\"Molecule overlap       : {len(overlap)}\")\n",
    "\n",
    "assert len(overlap) == 0, \"Molecule overlap detected between development and holdout!\"\n",
    "print()\n",
    "print(\"Molecule overlap = 0  --  confirmed clean split.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "DMOa5hQaJ_Rk",
   "metadata": {
    "id": "DMOa5hQaJ_Rk"
   },
   "source": [
    "## Section 12 — Define the Six Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "QvvqXLl7J_Rk",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QvvqXLl7J_Rk",
    "outputId": "09762438-adf4-4fae-a839-05e432c1e0b7"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Six models defined: ['XGBoost', 'Random Forest', 'SVM', 'Naive Bayes', 'RBF', 'ANN']\n"
     ]
    }
   ],
   "source": [
    "# We store *constructors* (functions that build a brand-new, untrained\n",
    "# model) rather than already-built model objects. This is required\n",
    "# because we need a completely fresh model for every one of the 10 CV\n",
    "# folds -- reusing one fitted object across folds would leak\n",
    "# information from earlier folds.\n",
    "model_builders = {\n",
    "    \"XGBoost\": lambda: XGBClassifier(\n",
    "        n_estimators=200, max_depth=6, learning_rate=0.1,\n",
    "        objective=\"binary:logistic\", eval_metric=\"logloss\",\n",
    "        random_state=RANDOM_STATE, n_jobs=-1, tree_method=\"hist\",\n",
    "    ),\n",
    "    \"Random Forest\": lambda: RandomForestClassifier(\n",
    "        n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1,\n",
    "    ),\n",
    "    \"SVM\": lambda: CalibratedClassifierCV(\n",
    "        LinearSVC(C=0.1, random_state=RANDOM_STATE, max_iter=5000),\n",
    "        method=\"sigmoid\", cv=3,\n",
    "    ),\n",
    "    \"Naive Bayes\": lambda: BernoulliNB(),\n",
    "    \"RBF\": lambda: Pipeline([\n",
    "        (\"rbf_features\", RBFSampler(gamma=0.01, n_components=500, random_state=RANDOM_STATE)),\n",
    "        (\"classifier\", SGDClassifier(loss=\"log_loss\", random_state=RANDOM_STATE, max_iter=1000)),\n",
    "    ]),\n",
    "    \"ANN\": lambda: MLPClassifier(\n",
    "        hidden_layer_sizes=(128, 64), activation=\"relu\", solver=\"adam\",\n",
    "        alpha=1e-4, early_stopping=False, max_iter=300, batch_size=64,\n",
    "        random_state=RANDOM_STATE,\n",
    "    ),\n",
    "}\n",
    "\n",
    "print(\"Six models defined:\", list(model_builders.keys()))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "F6_owIHtJ_Rk",
   "metadata": {
    "id": "F6_owIHtJ_Rk"
   },
   "source": [
    "## Section 13 — 10-Fold Molecule-Aware Cross-Validation\n",
    "\n",
    "**What:** For each of the six models, run 10-fold cross-validation on\n",
    "the development set only, where each fold's train/validation split is\n",
    "both **stratified** (keeps the Active/Inactive ratio similar across\n",
    "folds) and **group-aware** (a molecule's records never cross between\n",
    "train and validation within a fold)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "uJaEmTmJJ_Rk",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "uJaEmTmJJ_Rk",
    "outputId": "8079ac54-2da1-40e3-ae0e-08946c16f144"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Running 10-fold molecule-aware CV for: XGBoost ...\n",
      "  Mean CV F1 = 0.8834  (took 57.2s)\n",
      "Running 10-fold molecule-aware CV for: Random Forest ...\n",
      "  Mean CV F1 = 0.8757  (took 27.2s)\n",
      "Running 10-fold molecule-aware CV for: SVM ...\n",
      "  Mean CV F1 = 0.8818  (took 9.6s)\n",
      "Running 10-fold molecule-aware CV for: Naive Bayes ...\n",
      "  Mean CV F1 = 0.7893  (took 5.7s)\n",
      "Running 10-fold molecule-aware CV for: RBF ...\n",
      "  Mean CV F1 = 0.8634  (took 11.3s)\n",
      "Running 10-fold molecule-aware CV for: ANN ...\n",
      "  Mean CV F1 = 0.8833  (took 576.6s)\n",
      "\n",
      "All models completed 10-fold molecule-aware CV with zero group overlap in every fold.\n"
     ]
    }
   ],
   "source": [
    "def get_positive_class_scores(fitted_model, X_val):\n",
    "    \"\"\"Return a probability-like score for the positive class, used for ROC-AUC.\"\"\"\n",
    "    if hasattr(fitted_model, \"predict_proba\"):\n",
    "        return fitted_model.predict_proba(X_val)[:, 1]\n",
    "    return fitted_model.decision_function(X_val)\n",
    "\n",
    "\n",
    "def run_molecule_aware_cv(model_name, build_model, X_dev, y_dev, groups_dev, n_folds=N_CV_FOLDS):\n",
    "    \"\"\"Run stratified, molecule-aware CV for one model and return a dict of results.\"\"\"\n",
    "    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)\n",
    "\n",
    "    fold_accuracy, fold_precision, fold_recall = [], [], []\n",
    "    fold_f1, fold_roc_auc = [], []\n",
    "    group_overlap_count = 0\n",
    "\n",
    "    start_time = time.time()\n",
    "\n",
    "    for train_idx, val_idx in cv.split(X_dev, y_dev, groups=groups_dev):\n",
    "        # Safety check: no molecule should appear on both sides of this fold.\n",
    "        train_groups = set(groups_dev[train_idx])\n",
    "        val_groups = set(groups_dev[val_idx])\n",
    "        group_overlap_count += len(train_groups & val_groups)\n",
    "\n",
    "        X_train_fold, X_val_fold = X_dev[train_idx], X_dev[val_idx]\n",
    "        y_train_fold, y_val_fold = y_dev[train_idx], y_dev[val_idx]\n",
    "\n",
    "        model = build_model()\n",
    "        model.fit(X_train_fold, y_train_fold)\n",
    "\n",
    "        predictions = model.predict(X_val_fold)\n",
    "        scores = get_positive_class_scores(model, X_val_fold)\n",
    "\n",
    "        fold_accuracy.append(accuracy_score(y_val_fold, predictions))\n",
    "        fold_precision.append(precision_score(y_val_fold, predictions, zero_division=0))\n",
    "        fold_recall.append(recall_score(y_val_fold, predictions, zero_division=0))\n",
    "        fold_f1.append(f1_score(y_val_fold, predictions, zero_division=0))\n",
    "        fold_roc_auc.append(roc_auc_score(y_val_fold, scores))\n",
    "\n",
    "    elapsed = time.time() - start_time\n",
    "\n",
    "    assert group_overlap_count == 0, f\"{model_name}: molecule overlap found inside CV folds!\"\n",
    "\n",
    "    return {\n",
    "        \"Model\": model_name,\n",
    "        \"Mean CV Accuracy\": np.mean(fold_accuracy),\n",
    "        \"Mean CV Precision\": np.mean(fold_precision),\n",
    "        \"Mean CV Recall\": np.mean(fold_recall),\n",
    "        \"Mean CV F1\": np.mean(fold_f1),\n",
    "        \"Std CV F1\": np.std(fold_f1),\n",
    "        \"Mean CV ROC-AUC\": np.mean(fold_roc_auc),\n",
    "        \"CV Time (s)\": elapsed,\n",
    "    }\n",
    "\n",
    "\n",
    "cv_results = []\n",
    "for model_name, build_model in model_builders.items():\n",
    "    print(f\"Running 10-fold molecule-aware CV for: {model_name} ...\")\n",
    "    result = run_molecule_aware_cv(model_name, build_model, X_dev, y_dev, groups_dev)\n",
    "    cv_results.append(result)\n",
    "    print(f\"  Mean CV F1 = {result['Mean CV F1']:.4f}  (took {result['CV Time (s)']:.1f}s)\")\n",
    "\n",
    "print()\n",
    "print(\"All models completed 10-fold molecule-aware CV with zero group overlap in every fold.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "utEM78xCJ_Rk",
   "metadata": {
    "id": "utEM78xCJ_Rk"
   },
   "source": [
    "## Section 14 — Model Comparison and Automatic Best-Model Selection\n",
    "\n",
    "**What:** Put all six models' CV results into one table, sort it by\n",
    "Mean CV F1, and pick the top row as the winner.\n",
    "\n",
    "**Why:** F1 is a balanced measure of precision and recall, which suits\n",
    "this dataset since Active/Inactive are not perfectly balanced. The\n",
    "selection is a plain, data-driven sort — **no model name is\n",
    "hard-coded as the winner** — whichever model actually performs best\n",
    "in the experiment is the one that gets selected.\n",
    "\n",
    "**Output:** The full comparison table, and a clear statement of which\n",
    "model was selected and why.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "FenI0w1iJ_Rl",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 341
    },
    "id": "FenI0w1iJ_Rl",
    "outputId": "eea6aa3b-2aaf-44b4-add2-ea496e4ff6c4"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Model comparison (sorted by Mean CV F1, descending):\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "           Model  Mean CV Accuracy  Mean CV Precision  Mean CV Recall  \\\n",
       "0        XGBoost          0.818074           0.831428        0.943366   \n",
       "1            ANN          0.824512           0.857985        0.910599   \n",
       "2            SVM          0.815243           0.828448        0.943522   \n",
       "3  Random Forest          0.794720           0.785930        0.989625   \n",
       "4            RBF          0.786794           0.810871        0.925781   \n",
       "5    Naive Bayes          0.715395           0.854398        0.735006   \n",
       "\n",
       "   Mean CV F1  Std CV F1  Mean CV ROC-AUC  CV Time (s)  \n",
       "0    0.883443   0.018860         0.848337    57.228517  \n",
       "1    0.883333   0.018593         0.849928   576.635464  \n",
       "2    0.881843   0.017401         0.850113     9.561519  \n",
       "3    0.875707   0.020181         0.844839    27.242030  \n",
       "4    0.863449   0.024309         0.803433    11.303843  \n",
       "5    0.789277   0.039539         0.778382     5.651667  "
      ],
      "text/html": [
       "\n",
       "  <div id=\"df-f108d8e0-72eb-4e49-a92a-53de2b577878\" class=\"colab-df-container\">\n",
       "    <div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean CV Accuracy</th>\n",
       "      <th>Mean CV Precision</th>\n",
       "      <th>Mean CV Recall</th>\n",
       "      <th>Mean CV F1</th>\n",
       "      <th>Std CV F1</th>\n",
       "      <th>Mean CV ROC-AUC</th>\n",
       "      <th>CV Time (s)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>XGBoost</td>\n",
       "      <td>0.818074</td>\n",
       "      <td>0.831428</td>\n",
       "      <td>0.943366</td>\n",
       "      <td>0.883443</td>\n",
       "      <td>0.018860</td>\n",
       "      <td>0.848337</td>\n",
       "      <td>57.228517</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ANN</td>\n",
       "      <td>0.824512</td>\n",
       "      <td>0.857985</td>\n",
       "      <td>0.910599</td>\n",
       "      <td>0.883333</td>\n",
       "      <td>0.018593</td>\n",
       "      <td>0.849928</td>\n",
       "      <td>576.635464</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SVM</td>\n",
       "      <td>0.815243</td>\n",
       "      <td>0.828448</td>\n",
       "      <td>0.943522</td>\n",
       "      <td>0.881843</td>\n",
       "      <td>0.017401</td>\n",
       "      <td>0.850113</td>\n",
       "      <td>9.561519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.794720</td>\n",
       "      <td>0.785930</td>\n",
       "      <td>0.989625</td>\n",
       "      <td>0.875707</td>\n",
       "      <td>0.020181</td>\n",
       "      <td>0.844839</td>\n",
       "      <td>27.242030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>RBF</td>\n",
       "      <td>0.786794</td>\n",
       "      <td>0.810871</td>\n",
       "      <td>0.925781</td>\n",
       "      <td>0.863449</td>\n",
       "      <td>0.024309</td>\n",
       "      <td>0.803433</td>\n",
       "      <td>11.303843</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Naive Bayes</td>\n",
       "      <td>0.715395</td>\n",
       "      <td>0.854398</td>\n",
       "      <td>0.735006</td>\n",
       "      <td>0.789277</td>\n",
       "      <td>0.039539</td>\n",
       "      <td>0.778382</td>\n",
       "      <td>5.651667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>\n",
       "    <div class=\"colab-df-buttons\">\n",
       "\n",
       "  <div class=\"colab-df-container\">\n",
       "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f108d8e0-72eb-4e49-a92a-53de2b577878')\"\n",
       "            title=\"Convert this dataframe to an interactive table.\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
       "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
       "  </svg>\n",
       "    </button>\n",
       "\n",
       "  <style>\n",
       "    .colab-df-container {\n",
       "      display:flex;\n",
       "      gap: 12px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert {\n",
       "      background-color: #E8F0FE;\n",
       "      border: none;\n",
       "      border-radius: 50%;\n",
       "      cursor: pointer;\n",
       "      display: none;\n",
       "      fill: #1967D2;\n",
       "      height: 32px;\n",
       "      padding: 0 0 0 0;\n",
       "      width: 32px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert:hover {\n",
       "      background-color: #E2EBFA;\n",
       "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "      fill: #174EA6;\n",
       "    }\n",
       "\n",
       "    .colab-df-buttons div {\n",
       "      margin-bottom: 4px;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert {\n",
       "      background-color: #3B4455;\n",
       "      fill: #D2E3FC;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert:hover {\n",
       "      background-color: #434B5C;\n",
       "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
       "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
       "      fill: #FFFFFF;\n",
       "    }\n",
       "  </style>\n",
       "\n",
       "    <script>\n",
       "      const buttonEl =\n",
       "        document.querySelector('#df-f108d8e0-72eb-4e49-a92a-53de2b577878 button.colab-df-convert');\n",
       "      buttonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "\n",
       "      async function convertToInteractive(key) {\n",
       "        const element = document.querySelector('#df-f108d8e0-72eb-4e49-a92a-53de2b577878');\n",
       "        const dataTable =\n",
       "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
       "                                                    [key], {});\n",
       "        if (!dataTable) return;\n",
       "\n",
       "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
       "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
       "          + ' to learn more about interactive tables.';\n",
       "        element.innerHTML = '';\n",
       "        dataTable['output_type'] = 'display_data';\n",
       "        await google.colab.output.renderOutput(dataTable, element);\n",
       "        const docLink = document.createElement('div');\n",
       "        docLink.innerHTML = docLinkHtml;\n",
       "        element.appendChild(docLink);\n",
       "      }\n",
       "    </script>\n",
       "  </div>\n",
       "\n",
       "\n",
       "  <div id=\"id_32eea787-18ce-44eb-86af-9d5378ee3bd0\">\n",
       "    <style>\n",
       "      .colab-df-generate {\n",
       "        background-color: #E8F0FE;\n",
       "        border: none;\n",
       "        border-radius: 50%;\n",
       "        cursor: pointer;\n",
       "        display: none;\n",
       "        fill: #1967D2;\n",
       "        height: 32px;\n",
       "        padding: 0 0 0 0;\n",
       "        width: 32px;\n",
       "      }\n",
       "\n",
       "      .colab-df-generate:hover {\n",
       "        background-color: #E2EBFA;\n",
       "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "        fill: #174EA6;\n",
       "      }\n",
       "\n",
       "      [theme=dark] .colab-df-generate {\n",
       "        background-color: #3B4455;\n",
       "        fill: #D2E3FC;\n",
       "      }\n",
       "\n",
       "      [theme=dark] .colab-df-generate:hover {\n",
       "        background-color: #434B5C;\n",
       "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
       "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
       "        fill: #FFFFFF;\n",
       "      }\n",
       "    </style>\n",
       "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('results_df')\"\n",
       "            title=\"Generate code using this dataframe.\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
       "       width=\"24px\">\n",
       "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
       "  </svg>\n",
       "    </button>\n",
       "    <script>\n",
       "      (() => {\n",
       "      const buttonEl =\n",
       "        document.querySelector('#id_32eea787-18ce-44eb-86af-9d5378ee3bd0 button.colab-df-generate');\n",
       "      buttonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "\n",
       "      buttonEl.onclick = () => {\n",
       "        google.colab.notebook.generateWithVariable('results_df');\n",
       "      }\n",
       "      })();\n",
       "    </script>\n",
       "  </div>\n",
       "\n",
       "    </div>\n",
       "  </div>\n"
      ],
      "application/vnd.google.colaboratory.intrinsic+json": {
       "type": "dataframe",
       "variable_name": "results_df",
       "summary": "{\n  \"name\": \"results_df\",\n  \"rows\": 6,\n  \"fields\": [\n    {\n      \"column\": \"Model\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"XGBoost\",\n          \"ANN\",\n          \"Naive Bayes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Mean CV Accuracy\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.04044451318406178,\n        \"min\": 0.7153946180969412,\n        \"max\": 0.8245119722261224,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.8180743541581273,\n          0.8245119722261224,\n          0.7153946180969412\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Mean CV Precision\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.0270964201837431,\n        \"min\": 0.7859295037402242,\n        \"max\": 0.8579853744749893,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.8314282916457746,\n          0.8579853744749893,\n          0.8543978290431916\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Mean CV Recall\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.08879764723567213,\n        \"min\": 0.7350058378436171,\n        \"max\": 0.9896250405107991,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.943365808519356,\n          0.910598898052427,\n          0.7350058378436171\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Mean CV F1\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.03683226833588393,\n        \"min\": 0.7892765732897598,\n        \"max\": 0.8834425304413027,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.8834425304413027,\n          0.8833333664598746,\n          0.7892765732897598\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Std CV F1\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.008378182891246727,\n        \"min\": 0.0174008752352148,\n        \"max\": 0.039538632356547046,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.01886037284745929,\n          0.01859284790179183,\n          0.039538632356547046\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Mean CV ROC-AUC\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.030738415193647887,\n        \"min\": 0.7783817802108577,\n        \"max\": 0.8501131836365754,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.8483373953714892,\n          0.8499277704118817,\n          0.7783817802108577\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"CV Time (s)\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 227.1442666871463,\n        \"min\": 5.651667356491089,\n        \"max\": 576.6354639530182,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          57.228516817092896,\n          576.6354639530182,\n          5.651667356491089\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
      }
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "\n",
      "Best model based on Mean CV F1: XGBoost\n"
     ]
    }
   ],
   "source": [
    "results_df = pd.DataFrame(cv_results).sort_values(\"Mean CV F1\", ascending=False).reset_index(drop=True)\n",
    "\n",
    "print(\"Model comparison (sorted by Mean CV F1, descending):\")\n",
    "display(results_df)\n",
    "\n",
    "# The winner is read directly from the top row of the sorted table --\n",
    "# this line is the ONLY place the \"best model\" is decided, and it is\n",
    "# decided by the numbers above, not by us naming a model.\n",
    "best_model_name = results_df.iloc[0][\"Model\"]\n",
    "\n",
    "print()\n",
    "print(f\"Best model based on Mean CV F1: {best_model_name}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "AKslh84sJ_Rl",
   "metadata": {
    "id": "AKslh84sJ_Rl"
   },
   "source": [
    "## Section 15 — Train on Full Development Set, Evaluate Once on Holdout\n",
    "\n",
    "**What:** Train a fresh copy of the selected model on the **entire**\n",
    "70% development set, then evaluate it exactly once on the untouched\n",
    "30% holdout set.\n",
    "\n",
    "**Why:** The holdout set was never used for model selection or CV —\n",
    "this is the first and only time it is touched. This gives an honest\n",
    "estimate of how the chosen model performs on molecules it has never\n",
    "seen in any form.\n",
    "\n",
    "**Output:** Accuracy, Precision, Recall, F1, ROC-AUC and the confusion\n",
    "matrix on the holdout set.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "bNyhyGDZJ_Rl",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "bNyhyGDZJ_Rl",
    "outputId": "9fe06a58-eb3c-4d6a-bbc4-d11d29c0b420"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Training the selected model (XGBoost) on the full development set...\n",
      "\n",
      "Final holdout results (untouched 30% set, evaluated once):\n",
      "  Accuracy  : 0.8089\n",
      "  Precision : 0.8202\n",
      "  Recall    : 0.9388\n",
      "  F1        : 0.8755\n",
      "  ROC-AUC   : 0.8402\n",
      "\n",
      "Confusion matrix (rows = actual, columns = predicted; order = [Inactive, Active]):\n",
      "[[ 275  296]\n",
      " [  88 1350]]\n",
      "\n",
      "Note: this holdout set was not used to choose the model above --\n",
      "model selection (Section 14) used only the development-set CV results.\n"
     ]
    }
   ],
   "source": [
    "print(f\"Training the selected model ({best_model_name}) on the full development set...\")\n",
    "\n",
    "best_model = model_builders[best_model_name]()\n",
    "best_model.fit(X_dev, y_dev)\n",
    "\n",
    "holdout_predictions = best_model.predict(X_holdout)\n",
    "holdout_scores = get_positive_class_scores(best_model, X_holdout)\n",
    "\n",
    "holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)\n",
    "holdout_precision = precision_score(y_holdout, holdout_predictions, zero_division=0)\n",
    "holdout_recall = recall_score(y_holdout, holdout_predictions, zero_division=0)\n",
    "holdout_f1 = f1_score(y_holdout, holdout_predictions, zero_division=0)\n",
    "holdout_roc_auc = roc_auc_score(y_holdout, holdout_scores)\n",
    "holdout_confusion = confusion_matrix(y_holdout, holdout_predictions)\n",
    "\n",
    "print()\n",
    "print(\"Final holdout results (untouched 30% set, evaluated once):\")\n",
    "print(f\"  Accuracy  : {holdout_accuracy:.4f}\")\n",
    "print(f\"  Precision : {holdout_precision:.4f}\")\n",
    "print(f\"  Recall    : {holdout_recall:.4f}\")\n",
    "print(f\"  F1        : {holdout_f1:.4f}\")\n",
    "print(f\"  ROC-AUC   : {holdout_roc_auc:.4f}\")\n",
    "print()\n",
    "print(\"Confusion matrix (rows = actual, columns = predicted; order = [Inactive, Active]):\")\n",
    "print(holdout_confusion)\n",
    "print()\n",
    "print(\"Note: this holdout set was not used to choose the model above --\")\n",
    "print(\"model selection (Section 14) used only the development-set CV results.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eYKvn-N5J_Rl",
   "metadata": {
    "id": "eYKvn-N5J_Rl"
   },
   "source": [
    "## Section 16 — Save the Final Model and Metadata\n",
    "## Comparision graph of all models and confusion matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "NAGluvNWJ_Rl",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "NAGluvNWJ_Rl",
    "outputId": "6c881a36-58d7-49f4-f398-d74ab6ad3b25"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Saved best_model.pkl\n",
      "Saved metadata.json\n",
      "\n",
      "{\n",
      "  \"project_name\": \"Bioactive Molecule Prediction Using Extreme Gradient Boosting\",\n",
      "  \"selected_model\": \"XGBoost\",\n",
      "  \"feature_type\": \"Morgan fingerprint (ECFP4-style)\",\n",
      "  \"fingerprint_radius\": 2,\n",
      "  \"fingerprint_bits\": 2048,\n",
      "  \"dataset_row_counts\": {\n",
      "    \"raw_rows\": 7979,\n",
      "    \"cleaned_rows\": 6839,\n",
      "    \"development_rows\": 4830,\n",
      "    \"holdout_rows\": 2009\n",
      "  },\n",
      "  \"development_holdout_split\": {\n",
      "    \"holdout_fraction_target\": 0.3,\n",
      "    \"holdout_fraction_actual\": 0.2937563971340839,\n",
      "    \"split_type\": \"molecule-aware (GroupShuffleSplit)\",\n",
      "    \"molecule_overlap\": 0\n",
      "  },\n",
      "  \"cross_validation\": {\n",
      "    \"folds\": 10,\n",
      "    \"type\": \"StratifiedGroupKFold (molecule-aware, stratified)\",\n",
      "    \"selection_metric\": \"Mean CV F1\"\n",
      "  },\n",
      "  \"cv_results\": [\n",
      "    {\n",
      "      \"Model\": \"XGBoost\",\n",
      "      \"Mean CV Accuracy\": 0.81807435 ...\n"
     ]
    }
   ],
   "source": [
    "joblib.dump(best_model, \"best_model.pkl\")\n",
    "print(\"Saved best_model.pkl\")\n",
    "\n",
    "metadata = {\n",
    "    \"project_name\": \"Bioactive Molecule Prediction Using Extreme Gradient Boosting\",\n",
    "    \"selected_model\": best_model_name,\n",
    "    \"feature_type\": \"Morgan fingerprint (ECFP4-style)\",\n",
    "    \"fingerprint_radius\": FINGERPRINT_RADIUS,\n",
    "    \"fingerprint_bits\": FINGERPRINT_BITS,\n",
    "    \"dataset_row_counts\": {\n",
    "        \"raw_rows\": int(rows_before),\n",
    "        \"cleaned_rows\": int(len(v2_df)),\n",
    "        \"development_rows\": int(len(X_dev)),\n",
    "        \"holdout_rows\": int(len(X_holdout)),\n",
    "    },\n",
    "    \"development_holdout_split\": {\n",
    "        \"holdout_fraction_target\": HOLDOUT_FRACTION,\n",
    "        \"holdout_fraction_actual\": len(X_holdout) / len(X),\n",
    "        \"split_type\": \"molecule-aware (GroupShuffleSplit)\",\n",
    "        \"molecule_overlap\": 0,\n",
    "    },\n",
    "    \"cross_validation\": {\n",
    "        \"folds\": N_CV_FOLDS,\n",
    "        \"type\": \"StratifiedGroupKFold (molecule-aware, stratified)\",\n",
    "        \"selection_metric\": \"Mean CV F1\",\n",
    "    },\n",
    "    \"cv_results\": results_df.to_dict(orient=\"records\"),\n",
    "    \"final_holdout_results\": {\n",
    "        \"accuracy\": holdout_accuracy,\n",
    "        \"precision\": holdout_precision,\n",
    "        \"recall\": holdout_recall,\n",
    "        \"f1\": holdout_f1,\n",
    "        \"roc_auc\": holdout_roc_auc,\n",
    "        \"confusion_matrix\": holdout_confusion.tolist(),\n",
    "    },\n",
    "    \"random_state\": RANDOM_STATE,\n",
    "}\n",
    "\n",
    "with open(\"metadata.json\", \"w\") as f:\n",
    "    json.dump(metadata, f, indent=2)\n",
    "\n",
    "print(\"Saved metadata.json\")\n",
    "print()\n",
    "print(json.dumps(metadata, indent=2)[:800], \"...\")\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# ============================================================\n",
    "# Single-Plot Comparison of All 6 Models and Evaluation Metrics\n",
    "# ============================================================\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "metrics = [\n",
    "    \"Mean CV Accuracy\",\n",
    "    \"Mean CV Precision\",\n",
    "    \"Mean CV Recall\",\n",
    "    \"Mean CV F1\",\n",
    "    \"Mean CV ROC-AUC\"\n",
    "]\n",
    "\n",
    "metric_labels = [\n",
    "    \"Accuracy\",\n",
    "    \"Precision\",\n",
    "    \"Recall\",\n",
    "    \"F1\",\n",
    "    \"ROC-AUC\"\n",
    "]\n",
    "\n",
    "models_plot = results_df[\"Model\"].tolist()\n",
    "\n",
    "x = np.arange(len(models_plot))\n",
    "width = 0.15\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "\n",
    "for i, (metric, label) in enumerate(zip(metrics, metric_labels)):\n",
    "    values = results_df[metric].values\n",
    "\n",
    "    plt.bar(\n",
    "        x + (i - 2) * width,\n",
    "        values,\n",
    "        width,\n",
    "        label=label\n",
    "    )\n",
    "\n",
    "    # Display values above bars\n",
    "    for j, value in enumerate(values):\n",
    "        plt.text(\n",
    "            x[j] + (i - 2) * width,\n",
    "            value + 0.008,\n",
    "            f\"{value:.3f}\",\n",
    "            ha=\"center\",\n",
    "            va=\"bottom\",\n",
    "            fontsize=8\n",
    "        )\n",
    "\n",
    "plt.title(\n",
    "    \"Comparison of Six Machine Learning Models\",\n",
    "    fontsize=15\n",
    ")\n",
    "\n",
    "plt.xlabel(\"Machine Learning Model\", fontsize=12)\n",
    "plt.ylabel(\"Cross-Validation Score\", fontsize=12)\n",
    "\n",
    "plt.xticks(x, models_plot, rotation=20)\n",
    "plt.ylim(0, 1.08)\n",
    "\n",
    "plt.legend(\n",
    "    title=\"Evaluation Metrics\",\n",
    "    loc=\"lower right\"\n",
    ")\n",
    "\n",
    "plt.grid(\n",
    "    axis=\"y\",\n",
    "    alpha=0.3\n",
    ")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 375
    },
    "id": "lS5UBTP5ctC0",
    "outputId": "f540a081-cd17-47ab-e737-609d90ee1ce7"
   },
   "id": "lS5UBTP5ctC0",
   "execution_count": 35,
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 1400x700 with 1 Axes>"
      ],
      "image/png": "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dDACAwMNSYanp6exf//+dH06dOhgSDIKFixoREREGG3atDGKFi1q1KxZ06hTp06G79nvv/9uFC9e3JBkVK5c2Wjbtq3x+OOPG05OToajo6PxzTffWLU/evSoIclo0KCB1fbPPvvMkGQ4OjoaTzzxhNGxY0fjqaeesnz3AgDwsOG3GwAAD4j69esbkow5c+bcc98BAwYYkoxGjRoZiYmJlu379+83PDw8DEnGjz/+aNUnLTHg4+NjlRDYv3+/4erqahQsWNAICAgwpk6datl348YNo3HjxoYk44svvrA6XlpyQZLx+eefW7abzWZj+PDhhiSjRo0aVn3OnTtnrFixwjCbzVbb//rrLyMgIMBwcHBIlyC4/Tz/TAb8c2y3e+mllwxJxocffpiu/f79+42///7b8nrLli2Gg4ODUbhwYWPTpk2W7devXzeef/55Q5LRrl07q2OkJeIkGRMnTrTaN2jQIEOS0aVLlwzjzUjLli0NScaLL75o3Lhxw7J93bp1RsGCBQ1HR0dj586dVn0yS6beyaVLlyw/I66ursbzzz9vTJ482di4caPVef8ps3OdOXPG8PDwMJycnIyNGzcahmEYy5cvN0wmk1G2bFlLcj0rbk/cHj582JBkdOvWzbL/t99+MyQZr732mmEYRqaJ259++indea9fv2706tXLkGSMHj3aat+hQ4cMNzc3w8nJyfjqq6+s9pnNZmP58uXG9evX08VZsGBB4/fff7dsT0pKMurVq2dIMmbOnGl1nLslbu/lWO+//74hyWjYsKFx6tQpy/YbN24YL7/8siHJGD58uJFV95K4bdGihSHJGDhwoNV7cvr0aeORRx4xJBk///yzVZ97/TzSEreSjDZt2hjXrl3LNOaqVatavQd//fWXUaxYsXSJT8O4c+L2Xo6VmppqSYa+++67VseaMWOG5XjZTdyuWLEi3fuSdsElLXGdUeI2NjbWKFCggOHo6Gj88MMPVvGmfSc98sgjVn2++eYby8Wi29+XM2fOGEFBQZax3L4vISHB8Pb2NhwdHY25c+daHW/r1q1G8eLFjcKFCxvx8fGW7Zklbv39/Q2TyWRs3brVarvZbL6n7zUAAB4UJG4BAHhAVKlSxZBkLFu27J76XblyxShQoIDh4OCQ4eqpSZMmGZKMJ5980mp7WmJgxowZ6fq0adPGkGQ89thj6fb98MMP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cOBANW/eXFL2Po+8Lu3ixJNPPqkRI0bo66+/VrVq1fTnn39q69at6t+/vyZPnpzrtVn9/f0VFRWlyMhIRUREqH79+vLz89OOHTt08OBBeXp6Wlbvp+nYsaO+++47LVy4UNWqVVOTJk3k5OSk1atXq3z58qpTp442bdpk1adYsWL64YcfFBERod69e+udd95RUFCQihcvrtOnT2vHjh26dOmSvvvuOwUFBWUab3JysgYOHKihQ4eqZs2aCggIUHJysrZu3aq4uDgFBASoV69eOfJeAQBgL6y4BQDgAVO/fn0dOXJEH374oWrVqqXdu3drwYIF2rBhgwICAvTWW2/p8OHDatKkiaVPkSJF9Ouvv2r06NEqVaqUlixZonXr1umRRx7RvHnzNHHixFyL/4UXXtCSJUvk6OioH374QSdOnNCzzz6rjRs3KiwszKptp06d9N///ld16tRRTEyMfv75Z/n4+Gj16tVq1aqVTeN6++231b17d5lMJq1atUo//vijrl27ph49eigmJiZd2YPXX39dS5cuVYMGDbR161YtXrzYsrpx8+bN8vT0tGl8/1SmTBlt3bpV//73v+Xk5KTFixdr+/btatKkiZYvX64hQ4bY5Dzdu3fXt99+q549e6pEiRLavHmzFi5cqB07dqhatWp688039ccffyg4OPiOx0l7yv2VK1f04YcfpnsQ0YsvvqiOHTtq7969ubqysmTJktq6datefPFFJScna8mSJTp58qTefvvtDG8hT9OhQwdt27ZNnTt3VkJCghYtWqTt27fL399fH330kQoXLpxrY7gbBwcHffnll/rhhx/01FNP6ejRo1q0aJHWr18vNzc3vfrqq1lKvP/T5s2bM/1z4sQJSbfKjfz++++aNGmSqlWrpp07d+rbb7/V7t27Vb58eX3wwQcaOnSo5ZjZ/Tzyurp16+r3339Xy5YtdfToUS1ZskTOzs766aefVLduXUm2fzhcVnTp0kXr1q1Ty5YttX//fn377be6du2a+vbtq+3bt2f4wLB58+bpvffeU5kyZbRs2TJt2rRJL774olavXp3pxYo6depoz549GjZsmNzd3fXrr7/q+++/1/Hjx9WgQQPNmjXLUoIkM4ULF9bkyZMVERGhs2fPasmSJVq9erWKFy+u0aNHa/v27XZ5DwEAyEkm40EsFAUAAB44kZGRmj17ttas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lVC5cmWZzWa98cYbeuONN+Tm5iZHR0erNiaTSQkJCVk+5rlz55SamipPT0+r7Z6enjpw4ECGff766y+tXr1anTp10k8//aQjR46oX79+unnzpt56660M+9y4cUM3btywvE5MTJQkmc1mmc3mLMf7IHOQYbdzm7N3rcAmHOx4bkkyHOx4fpMdP/OHbF4xf3KfXeeOxPyxIeZP7suvv3sk5o8tMX/sgPljM/acOxLzxy74t5vNMH/sg/nz8LuXsWYrcduuXTuZTKbsdLUps9ksDw8Pff7553J0dFTNmjV18uRJffDBB5kmbseNG6fRo0en23727Fldv349p0POE6oWt99EjHcOsdu5KzmVttu5JSmpsv2+hNxK228Venx8vN3OnROYP7nPnnNHYv7YEvMn9+XX3z0S88eWmD+5j/ljO/acOxLzxx74t5vtMH/sg/nz8Lt8+XKW22YrcTtr1qzsdLujUqVKydHRUWfOnLHafubMGXl5eWXYx9vbW87OzlarfatWrarTp08rOTlZLi4u6fq89tprGjJkiOV1YmKi/Pz8VLp0abm7u9toNHnb/ov2S7p7uO2227kPl/S327klqdDBFLud+7qX490b5ZCMHiz4IGP+5D57zh2J+WNLzJ/cl19/90jMH1ti/uQ+5o/t2HPuSMwfe+DfbrbD/LEP5s/Dz83NLctts5W4zQkuLi6qWbOmVq1apdatW0u6taJ21apVGjBgQIZ96tevr3nz5slsNsvhf0vJDx06JG9v7wyTtpLk6uoqV1fXdNsdHBwsx3jYmWW/L18H2e/KkdmO55Ykkz2X/Rt2/MwfsnnF/Ml9dp07EvPHhpg/uS+//u6RmD+2xPyxA+aPzdhz7kjMH7vg3242w/yxD+bPw+9exprtdyUxMVGjR49W7dq15enpKU9PT9WuXVtjxoyx1I29V0OGDNH06dM1e/Zs7d+/X3379lVSUpK6desmSXrppZf02muvWdr37dtXFy5c0MCBA3Xo0CH997//1dixY9W/f//sDgsAAAAAAAAA7C5bK27//vtvPf744zp69KiqVKmi+vXrS5IOHjyoUaNG6csvv9S6devk7e19T8dt3769zp49qzfffFOnT59WjRo1tGzZMssDy2JjY62y0n5+flq+fLkGDx6skJAQlSlTRgMHDtTw4cOzMywAAAAAAAAAyBOylbgdPny4Tp8+raVLl6pFixZW+37++Wc9//zzGjFihGbPnn3Pxx4wYECmpRHWrl2bblvdunW1adOmez4PAAAAAAAAAORV2SqVsGzZMg0aNChd0laSnn76af3rX//STz/9dN/BAQAAAAAAAEB+lK3EbVJSkqV8QUa8vLyUlJSU7aAAAAAAAAAAID/LVuK2WrVq+vrrr5WcnJxu382bN/X111+rWrVq9x0cAAAAAAAAAORH2a5x2759e9WuXVv9+vVTYGCgpFsPJ5s2bZp2796t+fPn2zRQAAAAAAAAAMgvspW4ff7555WUlKQRI0aoT58+MplMkiTDMOTh4aEvvvhCzz33nE0DBQAAAAAAAID8IluJW0mKjIxU586dtW3bNh0/flySVLZsWT3yyCNycsr2YQEAAAAAAAAg37uvDKuTk5Pq1KmjOnXq2CoeAAAAAAAAAMj3spS4/e233yRJTzzxhNXru0lrDwAAAAAAAADIuiwlbhs2bCiTyaRr167JxcXF8jozhmHIZDIpNTXVZoECAAAAAAAAQH6RpcTtmjVrJEkuLi5WrwEAAAAAAAAAtpelxG2DBg3u+BoAAAAAAAAAYDsO2enUuHFjrVq1KtP9a9asUePGjbMdFAAAAAAAAADkZ9lK3K5du1ZnzpzJdH98fLx+/fXXbAcFAAAAAAAAAPlZthK3ku74cLIjR46oSJEi2T00AAAAAAAAAORrWapxK0mzZ8/W7NmzLa/feecdTZ8+PV27S5cuaffu3WrRooVtIgQAAAAAAACAfCbLidurV6/q7NmzlteXL1+Wg4P1gl2TyaRChQqpT58+evPNN20XJQAAAAAAAADkI1lO3Pbt21d9+/aVJJUrV04TJ05Uq1atciwwAAAAAAAAAMivspy4vd3Ro0dtHQcAAAAAAAAA4H+ylbi93eXLl5WQkCCz2Zxun7+///0eHgAAAAAAAADynWwnbqdOnaoJEybor7/+yrRNampqdg8PAAAAAAAAAPmWw92bpDdt2jT1799fFStW1DvvvCPDMDRo0CCNGDFCXl5eCg0N1cyZM20dKwAAAAAAAADkC9lK3H766adq1qyZfv75Z/Xq1UuS9Mwzz+jdd9/VH3/8ocuXL+v8+fM2DRQAAAAAAAAA8otsJW7//PNPRURESJKcnZ0lScnJyZKkokWLqkePHpoyZYqNQgQAAAAAAACA/CVbiduiRYsqJSVFkuTu7q6CBQsqLi7Osr9IkSI6ffq0bSIEAAAAAAAAgHwmW4nboKAg7dq1y/K6Tp06mjp1qk6ePKm4uDhFRUUpMDDQZkECAAAAAAAAQH7ilJ1OnTt31rRp03Tjxg25urpq9OjRevLJJ+Xv7y/pVvmERYsW2TRQAAAAAAAAAMgvspW47datm7p162Z5Xb9+fe3bt08//vijHB0d1bRpU1bcAgAAAAAAAEA2ZStxm5Hy5ctr4MCBtjocAAAAAAAAAORb2apxCwAAAAAAAADIOVlacevg4CCTyXTPB09NTb3nPgAAAAAAAACQ32Upcfvmm2+mS9x+99132rdvn5o1a6bKlStLkg4cOKBffvlFQUFBat26tc2DBQAAAAAAAID8IEuJ21GjRlm9/vzzzxUfH6+9e/dakrZp9u/fr8aNG8vHx8dmQQIAAAAAAABAfpKtGrcffPCBBgwYkC5pK0lVq1bVgAED9P777993cAAAAAAAAACQH2UrcXvixAk5Oztnut/Z2VknTpzIdlAAAAAAAAAAkJ9lK3EbFBSkKVOm6OTJk+n2nThxQlOmTFFwcPB9BwcAAAAAAAAA+VGWatz+08cff6xmzZopMDBQbdq0UcWKFSVJhw8f1vfffy/DMDR37lybBgoAAAAAAAAA+UW2ErePPfaYNm/erDfeeEPfffedrl27JkkqUKCAmjVrptGjR7PiFgAAAAAAAACyKVuJW+lWuYTvvvtOZrNZZ8+elSSVLl1aDg7Zqr4AAAAAAAAAAPifbCdu0zg4OMjT09MWsQAAAAAAAAAAlMXE7ZgxY2QymTRy5Eg5ODhozJgxd+1jMpn0xhtv3HeAAAAAAAAAAJDfZClxO2rUKJlMJg0fPlwuLi4aNWrUXfuQuAUAAAAAAACA7MlS4tZsNt/xNQAAAAAAAADAdniSGAAAAAAAAADkMSRuAQAAAAAAACCPyVKphHLlyslkMt3TgU0mk/78889sBQUAAAAAAAAA+VmWErcNGjS458QtAAAAAAAAACB7spS4nTVrVg6HAQAAAAAAAABIQ41bAAAAAAAAAMhjsrTiNjM3b97UgQMHlJCQILPZnG7/E088cT+HBwAAAAAAAIB8KVuJW7PZrNdee01TpkzR1atXM22Xmpqa7cAAAAAAAAAAIL/KVqmEsWPH6oMPPlDnzp315ZdfyjAMjR8/XtOmTVNISIhCQ0O1fPlyW8cKAAAAAAAAAPlCthK3s2bN0gsvvKCpU6eqefPmkqSaNWuqZ8+e2rx5s0wmk1avXm3TQAEAAAAAAAAgv8hW4vbEiRNq3LixJMnV1VWSdP36dUmSi4uLOnfurDlz5tgoRAAAAAAAAADIX7KVuC1ZsqSuXLkiSSpcuLDc3d31119/WbW5ePHi/UcHAAAAAAAAAPlQth5OFhYWpq1bt1peN2rUSJ988onCwsJkNps1adIkhYaG2ixIAAAAAAAAAMhPsrXitmfPnrpx44Zu3LghSXr33Xd16dIlPfHEE2rQoIESExP10Ucf2TRQAAAAAAAAAMgvsrzidujQoerSpYtCQ0P17LPP6tlnn7Xsq1atmv7880+tXbtWjo6OqlevnkqUKJEjAQMAAAAAAADAwy7LK24nTJig8PBwVa9eXePGjdPx48et9hctWlTPPvusWrZsSdIWAAAAAAAAAO5DlhO3hw4d0ptvvimz2ayRI0eqfPnyeuyxxzRt2jSdP38+J2MEAAAAAAAAgHwly4nbihUr6q233tL+/fu1bds2DRo0SMePH1e/fv3k4+OjVq1aaf78+bp27VpOxgsAAAAAAAAAD71sPZwsPDxcH330kWJjY7Vq1Sq99NJL2rBhgzp27ChPT0+99NJLWr58ua1jBQAAAAAAAIB8IVuJ2zQmk0mNGjXS9OnTdfr0aX3//feqX7++5s6dq2eeecZWMQIAAAAAAABAvuJki4MkJydr6dKlmjdvntauXStJ8vT0tMWhAQAAAAAAACDfyXbi1jAMrVq1SvPmzdN3332nhIQEFSlSRO3bt1fnzp3VuHFjW8YJAAAAAAAAAPnGPSdut2zZonnz5mnBggU6c+aMnJyc1KxZM3Xu3FmtWrWSm5tbTsQJAAAAAAAAAPlGlhO3b775pr7++mv99ddfMgxD9erV0xtvvKH27durRIkSORkjAAAAAAAAAOQrWU7cvvPOO6pSpYrGjBmjTp06KSAgIAfDAgAAAAAAAID8K8uJ2+3btyssLCzDfRcvXlS7du300UcfZdoGAAAAAAAAAJA1DllteKeEbHJystauXauLFy/aJCgAAAAAAAAAyM+ynLgFAAAAAAAAAOQOErcAAAAAAAAAkMfYJHFboEABde3aVT4+PrY4HAAAAAAAAADka1l+ONmduLu7Kzo62haHAgAAAAAAAIB8L1srbmNjY7V+/Xqrbbt27dJLL72k9u3b6/vvv7dFbAAAAAAAAACQL2Vrxe2//vUvXblyRStXrpQknTlzRo0aNVJycrKKFCmib7/9VgsXLlTbtm1tGiwAAAAAAAAA5AfZWnG7ZcsWPfXUU5bXX375pa5du6Zdu3bp5MmTatKkiT788EObBQkAAAAAAAAA+Um2ErcXLlyQh4eH5fXSpUvVoEEDVahQQQ4ODmrbtq0OHDhgsyABAAAAAAAAID/JVuK2dOnSOn78uCTp0qVL+r/27jI8iuv/+/hnN0qQJFiCBEJwCBqkSNAgxV2LFilWQUppi7c4tDgFihWXAi20FIr0hzvFnQDFXRKI7dwPuNl/twkesgu8X9eVC+bMmSO7OdnZ75w5s23bNlWqVMm6Pzo6WtHR0fHTQgAAAAAAAAB4x7zUGrchISEaM2aMkiVLpg0bNshisahWrVrW/YcPH5afn198tREAAAAAAAAA3ikvFbgdMmSIjh8/ru7du8vV1VUjRoxQpkyZJEkRERFauHChmjRpEq8NBQAAAAAAAIB3xUsFbn18fLR582bduXNHiRIlkqurq3WfxWLR2rVrmXELAAAAAAAAAC/ppQK3j3l6esZKS5QokfLly/cqxQIAAAAAAADAO+2lHk62du1aDR8+3CZt2rRpypAhg3x8fPTZZ58pJiYmXhoIAAAAAAAAAO+alwrc9uvXT3///bd1+8CBA2rfvr1SpUqlMmXKaMyYMRoxYkS8NRIAAAAAAAAA3iUvFbg9cuSIChUqZN3+6aeflCxZMm3cuFELFixQ27ZtNWvWrHhrJAAAAAAAAAC8S14qcBsWFqZkyZJZt1etWqXKlSvLw8NDklS4cGGdPXs2floIAAAAAAAAAO+Ylwrc+vn5aefOnZKkkydP6uDBg6pYsaJ1/82bN+Xm5hY/LQQAAAAAAACAd4zzyxzUtGlTDRgwQBcuXNChQ4fk7e2tmjVrWvfv3r1b2bJli7dGAgAAAAAAAMC75KUCt1999ZUiIyP122+/KUOGDJoxY4a8vLwkPZptu2HDBn3yySfx2U4AAAAAAAAAeGe8VODW2dlZ3377rb799ttY+5InT67Lly+/csMAAAAAAAAA4F31UoHbf7t//77Onz8v6dHat0mSJHnlRgEAAAAAAADAu+ylHk4mSTt37lTZsmXl7e2twMBABQYGytvbW+XKldOuXbvis40AAAAAAAAA8E55qRm327dvV5kyZeTq6qo2bdooZ86ckqQjR45o3rx5KlWqlDZs2KAiRYrEa2MBAAAAAAAA4F3w0g8nS5cunTZt2iRfX1+bff369VOJEiX01Vdfac2aNfHSSAAAAAAAAAB4l7zUUgnbt29X+/btYwVtJcnHx0ft2rXTtm3bXrpR48ePl7+/v9zd3VW0aFHt2LHjuY6bP3++TCaTatWq9dJ1AwAAAAAAAIC9vVTg1mw2Kzo6+on7Y2JiZDa/3PK5CxYsUNeuXdW3b1/t2bNH+fLlU6VKlXT16tWnHhcaGqru3bsrODj4peoFAAAAAAAAAEfxUtHV4sWLa/z48Tp79mysfefOndOECRNUokSJl2rQqFGj1LZtW7Vq1Uq5cuXSpEmT5OHhoWnTpj3xmJiYGDVt2lT9+/dXQEDAS9ULAAAAAAAAAI7ipda4HTRokIKDg5UjRw7Vrl1b2bJlkyQdO3ZMy5cvl7OzswYPHvzC5UZGRmr37t3q1auXNc1sNiskJERbt2594nEDBgxQ6tSp9eGHH2rjxo1PrSMiIkIRERHW7bt370qSLBaLLBbLC7f5TWSWYbe6LS93rSBemO1YtyQZLzkLPV6Y7Piev2XjivGT8Ow6diTGTzxi/CS8d/WzR2L8xCfGjx0wfuKNPceOxPixC87d4g3jxz4YP2+/F+nrSwVuCxQooB07duirr77SL7/8ovDwcEmSh4eHKleurG+++Ua5cuV64XKvX7+umJgY+fj42KT7+Pjo6NGjcR6zadMm/fjjj9q3b99z1TF48GD1798/Vvq1a9f08OHDF27zmyint/0G4lWXvHarO6tzKrvVLUlh2e33R8g9VYzd6n7WMidvGsZPwrPn2JEYP/GJ8ZPw3tXPHonxE58YPwmP8RN/7Dl2JMaPPXDuFn8YP/bB+Hn73bt377nzvnDgNiIiQn/88Yf8/f21dOlSWSwWXbt2TZKUKlWql17b9mXcu3dPzZo105QpU5QyZcrnOqZXr17q2rWrdfvu3bvy8/NTqlSplCxZstfVVIdy5JbJbnWndt9vt7pPpMhgt7olKfGxJ68L/bo99HWyW92pU6e2W92vA+Mn4dlz7EiMn/jE+El47+pnj8T4iU+Mn4TH+Ik/9hw7EuPHHjh3iz+MH/tg/Lz93N3dnzvvCwduXV1dVb9+fY0ePVp58+aV2WyONUP2ZaVMmVJOTk66cuWKTfqVK1fk6+sbK/+pU6cUGhqq6tWrW9MeTzd2dnbWsWPHlDlzZptj3Nzc5ObmFqsss9mcoEFne7LIfn98zbLflSOLHeuWJJM9p/0bdnzP37JxxfhJeHYdOxLjJx4xfhLeu/rZIzF+4hPjxw4YP/HGnmNHYvzYBedu8YbxYx+Mn7ffi/T1hV8Vk8mkrFmz6vr16y966DO5uroqKChIa9eutaZZLBatXbtWxYoVi5U/R44cOnDggPbt22f9qVGjhsqWLat9+/bJz88v3tsIAAAAAAAAAK/bS61x++WXX6pr166qX7++smfPHq8N6tq1q1q0aKFChQqpSJEi+v777xUWFqZWrVpJkpo3b6506dJp8ODBcnd3V2BgoM3xXl5ekhQrHQAAAAAAAADeFC8VuN22bZtSpEihwMBAlSlTRv7+/kqUKJFNHpPJpNGjR79w2Q0bNtS1a9fUp08fXb58Wfnz59eqVausyzGcO3funZo+DQAAAAAAAODd81KB23Hjxln//+9lDf7tZQO3ktS5c2d17tw5zn0bNmx46rEzZsx4qToBAAAAAAAAwFG8VODWYu8HxQAAAAAAAADAW4w1BwAAAAAAAADAwTx34Pbhw4f66KOPNHbs2KfmGzNmjDp06KCoqKhXbhwAAAAAAAAAvIueO3A7efJkzZgxQ1WrVn1qvqpVq2r69OmaOnXqKzcOAAAAAAAAAN5Fzx24XbhwoerWrauAgICn5sucObPq16+vefPmvXLjAAAAAAAAAOBd9NyB2wMHDqhkyZLPlbd48eLav3//SzcKAAAAAAAAAN5lzx24jYyMlKur63PldXV1VURExEs3CgAAAAAAAADeZc8duE2bNq0OHjz4XHkPHjyotGnTvnSjAAAAAAAAAOBd9tyB25CQEM2aNUtXr159ar6rV69q1qxZqlChwis3DgAAAAAAAADeRc8duO3Zs6cePnyocuXKafv27XHm2b59u8qXL6+HDx+qR48e8dZIAAAAAAAAAHiXOD9vxoCAAC1cuFCNGzdW8eLFFRAQoDx58ihp0qS6d++eDh48qFOnTsnDw0Pz589X5syZX2e7AQAAAAAAAOCt9dyBW0mqWrWq9u/fr6FDh2rFihVatmyZdV/atGnVtm1bff755woICIjvdgIAAAAAAADAO+OFAreS5O/vr4kTJ2rixIm6d++e7t69q2TJkilp0qSvo30AAAAAAAAA8M554cDtvyVNmpSALQAAAAAAAADEs+d+OBkAAAAAAAAAIGEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB0PgFgAAAAAAAAAcDIFbAAAAAAAAAHAwBG4BAAAAAAAAwMEQuAUAAAAAAAAAB+Ns7wYAAAAAAAAAbwonk5NSuqSU+TXMh7SkiY73Mp+Xu6f95nc+fPjQbnXHNxcXFzk5OcVLWQRuAQAAAAAAgOfg7eytT/w/kaerp0wyxXv5MV/He5HPLY+bp93qPnPmjN3qfh28vLzk6+srk+nVfkcI3AIAAAAAAADPYJJJdXzqKF2ydEqcIrFeQ9xWfteM+C/0OYUlTmu3ulOkS2K3uuOTYRgKDw/X1atXJUlp0qR5pfII3AIAAAAAAADPkMQpiXImyykPLw+ZXV/PsgJuZvsFbiOdXe1Wt7u7u93qjm+JEiWSJF29elWpU6d+pWUTeDgZAAAAAAAA8AweTh5yNjnL5PQaptrireLh4SFJioqKeqVyCNwCAAAAAAAAz2B9GBlxWzzDq65t+xiBWwAAAAAAAABwMARuAQAAAAAAAAd39sIFeeTJo7+PHn3tdc1fNEdZ82R47fXYU2hoqEwmk/bt22fvpjwRgVsAAAAAAADgFXzV+SsFpgqM9dO+QXt7N+2ZclSqpHE//WSTVrN6HW1Zv/u11127YVX5+HtqyJAhsfZVrVpVJpNJ/fr1e+7yZsyYIS8vr+fK6+fnp0uXLikwMPC5y09ozvZuAAAAAAAAAPCmK1mupL4Z841Nmoubi51a82oSuSdSIvdECVJXurTpNWPGDH3xxRfWtAsXLmjt2rVKkybNa6kzMjJSrq6u8vX1fS3lxxdm3AIAAAAAAACvyNXNVSl9Utr8eHp5SpI+b/+5urXpZpM/KipKJbOX1PIFyyVJm9ZuUvnmzZWmeHGlL1lSdTp10unz559Y30/LlilN8eI2ab+sXSuPPHms26fPn1f9Ll3kX7q0UhUpopKNGmnd1q3W/ZVatdK5ixf1+bBh8vH3lI//o/bGtVTCjJ+mqkipfEqfNaWKlwvSop/n2+z38ffU7Pkz1bJdU/nn8NV7ZQpo1Zrfnvm6VShXSdevX9fmzZutaTNnzlTFihWVOnVqm7wRERHq3r270qVLp8SJE6to0aLasGGDJGnDhg1q1aqV7ty5I5PJZDNb19/fXwMHDlTz5s2VLFkytWvXLs6lEg4dOqRq1aopWbJkSpo0qYKDg3Xq1Clr+UWKFFHixInl5eWlEiVK6OzZs8/s36sgcAsAAAAAAAC8RlXrVtVfq/9S+P1wa9rm9Zv18MFDhVQNkSQ9CH+gLs2ba9P8+Vo5darMZrMaffKJLBbLS9d7PzxclYKDtXLqVG1dtEgVSpRQvS5ddP7SJUnSvO+/VzofH/Xu1EkHdhzXgR3H4yznt1W/6usBX6hD2876649tat6klT7p0VGbtvzPJt/I0UNVo2ptrV+1WeXLVFTHT9vq1u2bT22ji4urmjZtqunTp1vTZsyYodatW8fK27lzZ23dulXz58/X/v37Vb9+fVWuXFknTpxQ8eLF9f333ytZsmS6dOmSLl26pO7du1uPHTFihPLly6e9e/eqd+/escq+cOGCSpUqJTc3N61bt067d+9W69atFR0drejoaNWqVUulS5fW/v37tXXrVrVr104mk+mpfXtVLJUAAAAAAAAAvKK/Vv+lwhkL26S1/bSt2n3WTiXKlVAij0T687c/VaNBDUnSb0t+U5lKZZQ4SWJJUoXqFRRw2bAeO2nAAGUoVUpHTp1S7qxZX6pNebNnV97s2a3bfbt00a/r1mnF+vXq0KSJknt6ysnJSUkTJ1bq1D5PLGfClLFqWK+JWjVrK0nKHNBZu/fu1MQpY1WyeClrvob1mqhOzXqSpC8/76OpMyZp7749Klcm5KntbN26tYKDgzV69Gjt3r1bd+7cUbVq1WzWtz137pymT5+uc+fOKW3atJKk7t27a9WqVZo+fboGDRokT09PmUymOJdAKFeunLp1+79Zz6GhoTb7x48fL09PT82fP18uLo+WuMiWLZsk6ebNm9Y2Zc6cWZKUM2fOp/YpPhC4BQAAAAAAAF5R4ZKF1WdYH5s0T+9HSw84OzurUo1KWrl4pWo0qKHwsHCtX7VewycPt+Y9e+qs+vcfp5379+vG7dvWmbbnL1166cDt/fBwfTthglb973+6fP26oqOj9SAiQv9cvvxC5Zw4eUzNGre0SSsc9J6mTJ9ok5YrR27r/xN7JFbSpMl0/ca1Z5afL18+Zc2aVYsXL9b69evVrFkzOTvbhi0PHDigmJgYazD1sYiICKVIkeKZdRQqVOip+/ft26fg4GBr0PbfkidPrpYtW6pSpUqqUKGCQkJC1KBBg9e2Bu9jBG4BAAAAAACAV+Th4aEMARmeuL9qvapqVbOVbly7oa1/bZWbu5tKlCth3d/5g87KnDqNxvfrpzSpU8tisahQ7dqKjIqKszyz2SzDMGzSoqOjbbZ7jRihdVu3alD37srs56dE7u5q0rXrE8t8Vc7OtkFPk0zPvdRD69atNX78eB0+fFg7duyItf/+/ftycnLS7t275eTkZLMvSZIkzyw/ceLET92fKNHTH8Y2ffp0ffzxx1q1apUWLFigr7/+WmvWrNF77733zLpfFmvcAgAAAAAAAK9ZgSIF5JvOV6uWrdLKxStVsUZF6+zO2zdv68zJM+rZrp3KvveecgQE6Pbdu08tL6W3t+6FhSks/P/Wzf372DGbPNv27tUHNWuqZvnyCsyWTT4pU+rcxYs2eVxdXBTzjOBq1izZtWP3Npu0nbu3KVvWHM/s9/Nq0qSJDhw4oMDAQOXKlSvW/gIFCigmJkZXr15VlixZbH4eL43g6uqqmJiYl6o/b9682rhxo6KeEtQuUKCAevXqpS1btigwMFBz5859qbqeFzNuAQAAAAAAgFcUGRGp61eu26Q5OTvJO4W3dbtKnSpaOHOhzp46q2lLp1nTk3klk1dyL01bvFi+qVLp/KVL6v3990+tr3DevPJwd1ffMWPUsWlT7dy/X7OXL7fJkzljRi1fu1ZVypSRyWTSgHHjYs2AzZg2rTbt2qX3L1+Uq6ubUiSPvexAx3Yfq13nlsqTK69KlSyr1Wt/18pVv2rR7OWx8r4sb29vXbp0Kc6lCqRH6802bdpUzZs318iRI1WgQAFdu3ZNa9euVd68eVW1alX5+/vr/v37Wrt2rfLlyycPDw95eHg8V/2dO3fW2LFj1ahRI/Xq1Uuenp7atm2bihQpIldXV02ePFk1atRQ2rRpdezYMZ04cULNmzePt/7HhRm3AAAAAAAAwCvatG6TygSWsflpXs02sFe1XlWdOnZKqdOkVoGiBazpZrNZwycP197Dh1Wodm31HDZMg7p2fWp9yT099ePgwfpj40YVrlNHC3//XV916GCTZ2iPHvJOlkzlmjVTvc6dFVK8uPL/56FavTt10rmLF1W0VH7lKhgQZ11VKlXTN32GaMKUsSpVsahmzZ2u0cMnqESx4Bd5iZ7Jy8vrqUsaTJ8+Xc2bN1e3bt2UPXt21apVSzt37lSGDI+WqChevLg++ugjNWzYUKlSpdKwYcOeu+4UKVJo3bp1un//vkqXLq2goCBNmTJFLi4u8vDw0NGjR1W3bl1ly5ZN7dq1U6dOndS+fftX7vPTMOMWAAAAAAAAeAXfjvtW34779pn5MmfLrIPXDsa5r1jpYtrznxmz4QcOWP+fMV06m21JqlG+vGqUL2+T1rpePZtjfv/xR5v9HzVubLNdJF8+bV+yRPeS/t/6vI3qN1Wj+k1t8rVs1kYtm7V5Utd0JfROrLQTB849Mb8kLV2w8qn79+3bZ7Pt4uKi/v37q3///k88ZuLEiZo40fahaaGhobHy+fv7x1ojOG/evPrjjz/ibuvSpU9t6+vAjFsAAAAAAAAAcDAEbgEAAAAAAADAwRC4BQAAAAAAAAAHQ+AWAAAAAAAAABwMgVsAAAAAAAAAcDAEbgEAAAAAAADAwRC4BQAAAAAAAAAHQ+AWAAAAAAAAABwMgVsAAAAAAAAAcDAEbgEAAAAAAADAwTjbuwFxGT9+vIYPH67Lly8rX758Gjt2rIoUKRJn3ilTpmjWrFk6ePCgJCkoKEiDBg16Yn4AAAAAAADAnqqOCE3Q+o58kPGljtu5e4dq1K+kcqVDNGf6onhuFZ7F4WbcLliwQF27dlXfvn21Z88e5cuXT5UqVdLVq1fjzL9hwwY1btxY69ev19atW+Xn56eKFSvqwoULCdxyAAAAAAAA4O0xd+Esfdiivbbu2KLLVy7ZrR2RkZF2q9ueHC5wO2rUKLVt21atWrVSrly5NGnSJHl4eGjatGlx5p8zZ446duyo/PnzK0eOHJo6daosFovWrl2bwC0HAAAAAAAA3g5hYfe1fMVStfzgQ4WUraj5i+fY7P/jz99VqUYZZciWWjkLZFLLdk2t+yIiIjRwcB8VKJZLftlSqWjp/JqzYJYkaf6iOcqaJ4NNWcuWLZPJZLJu9+vXT/nz59fUqVOVKVMmubu7S5JWrVqlkiVLysvLSylSpFC1atV06tQpm7L++ecfNW7cWMmTJ1fixIlVqFAhbd++XaGhoTKbzdq1a5dN/u+//14ZM2aUxWJ59RctnjnUUgmRkZHavXu3evXqZU0zm80KCQnR1q1bn6uM8PBwRUVFKXny5HHuj4iIUEREhHX77t27kiSLxeKQb9DrYJZht7otdrxWYLbzdQrDbMf6TXZ8z9+yccX4SXh2HTsS4yceMX4S3rv62SMxfuIT48cOGD/xxp5jR2L82AXnbvGG8RObSaY4099Wy1cuVdbMWZUlc1bVq91QvQd8oU86dpPJZNKadX+oVfum+rRTd40d9YOiIiP15/rV1mM7d22v3Xt36tu+Q5UrV6DOnT+rmzdvPLEuwzBi/Xvy5EktWbJES5YskZOTkwzD0P379/XZZ58pb968un//vvr27avatWtr7969MpvNun//vkqXLq106dJp+fLl8vX11Z49exQTE6OMGTMqJCRE06ZNU1BQkLXu6dOnq0WLFjKZTNb6X5VhGDIMI85444v8rXCowO3169cVExMjHx8fm3QfHx8dPXr0ucro2bOn0qZNq5CQkDj3Dx48WP3794+Vfu3aNT18+PDFG/0Gyultvz++V13y2q3urM6p7Fa3JIVlt9+HuHuqGLvV/aRlTt5UjJ+EZ8+xIzF+4hPjJ+G9q589EuMnPjF+Eh7jJ/7Yc+xIjB974Nwt/jB+YkvunFwuJhe5yU1OJqcEbtWrsfz/GatPY3K2fc/nLvxJ9eo0kMnZUPny5fVpj7vaunOjShQP1vfjR6hWjbrq+fn/Tb4MzBsoydCp0yf1y8qlWjRvmUoHl5EkZQrw//+5DJmcDMlk2NQXE/No7EZHRz9qr8WiyMhI/fjjj0qVKpV1X82aNW3a+MMPPyht2rTav3+/AgMDNXv2bF27dk1btmyxTur09/e3Ht+yZUt17txZw4YNk5ubm/bu3asDBw5o8eLF1rrjQ3R0tCwWi27cuCEXFxebfffu3XvuchwqcPuqhgwZovnz52vDhg3WKdT/1atXL3Xt2tW6fffuXfn5+SlVqlRKlixZQjXVro7cst8VotTu++1W94kUGZ6d6TVKfCz+/gC8qIe+9vtASZ06td3qfh0YPwnPnmNHYvzEJ8ZPwntXP3skxk98YvwkPMZP/LHn2JEYP/bAuVv8YfzE5mv2VZQRpQhFyGw43OqjT2V+jsmKhsv/vecnT53Q3n27NX3SHBnRJjnJRTWr1dGcebNVvEgpHTp0QB80bCEjOvbvyYH9B+Tk5KRihUrGud+IMUmGyWafk9Ojsevs/ChUaTablTFjRqVJk8bm2BMnTqhv377avn27rl+/bp29evHiReXPn1/79+9XgQIFnjge69atq08++US//vqrGjVqpJ9++klly5ZVlixZnvn6vAhnZ2eZzWalSJEiVozySTHLOMuJ11a9opQpU8rJyUlXrlyxSb9y5Yp8fX2feuyIESM0ZMgQ/fnnn8qb98lXZdzc3OTm5hYr3Ww2y2zv23ETiMWOU/vNst+VV4sd65Ykkz1vmzHs+J6/ZeOK8ZPw7Dp2JMZPPGL8JLx39bNHYvzEJ8aPHTB+4o09x47E+LELzt3iDeMnNsPOy0ckpLkLZyk6Olr5ima3phmGITdXNw3uP/ypwUd390RPLdtsNsd6LR/Pdn28zq3JZFLixIlt1r2VpBo1aihjxoyaMmWK0qZNK4vFosDAQEVFRclkMsnDw8OmnP9yc3NT8+bNNWPGDNWtW1fz5s3T6NGjn5j/ZZlMJplMpjjjjS/yt8Kh/qq4uroqKCjI5sFijx80VqxYsSceN2zYMA0cOFCrVq1SoUKFEqKpAAAAAAAAwFsnOjpaC5fMV/+vv9Xa3zZZf9b9vlk+Pr5a+sti5cwRqI1b/orz+JzZc8lisWjL9k1x7k+RPKXu37+nsPAwa9q+ffue2a4bN27o2LFj+vrrr1W+fHnlzJlTt27dssmTN29e7du3Tzdv3nxiOW3atNGff/6pCRMmKDo6WnXq1Hlm3fbiUIFbSerataumTJmimTNn6siRI+rQoYPCwsLUqlUrSVLz5s1tHl42dOhQ9e7dW9OmTZO/v78uX76sy5cv6/79+/bqAgAAAAAAAPBGWr12le7cva0mDZopZ/ZcNj/VKtfQ3IU/qfsnPbX0l8UaNmqQjp88psNHD2nsxO8kSRn8Mqph3Sb67PPO+u2PFTp7PlSbt27U8hU/S5IKFghSokQeGjRsgELPntaS5Ys0Y8aMZ7bL29tbKVKk0OTJk3Xy5EmtW7fOZjlUSWrcuLF8fX1Vq1Ytbd68WadPn9aSJUu0detWa56cOXPqvffeU8+ePdW4cWMlSvT0GcL25HCB24YNG2rEiBHq06eP8ufPr3379mnVqlXWB5adO3dOly5dsuafOHGiIiMjVa9ePaVJk8b6M2LECHt1AQAAAAAAAHgjzV34k0qVKKNkyTxj7av6fk3t279XXl7emjphpv748zeVr1JS9ZpU196/d1vzDf1mlKq9X1Nf9O6mkuULq1uvjxUeHi5J8vZKrvHfTdbaDatVplJxLf1lsfr16/fMdpnNZs2fP1+7d+9WYGCgPvvsMw0fPtwmj6urq1avXq3UqVOrSpUqypMnj4YMGWJdQ/exDz/8UJGRkWrduvVLvEIJx6HWuH2sc+fO6ty5c5z7NmzYYLMdGhr6+hsEAAAAAAAAxJOV3f3jTA+4bP91dGf/uOCJ+wrmD9KV0DuSpNw5A1W1co0487m7u2tA70Ea0HtQnPurVKqmKpWqWbdTZ0ymtm3bWrf79esXZzA3JCREhw8ftkkzDNvXLGPGjFq8ePET+yBJFy5cUJ48eVS4cOGn5rM3h5txCwAAAAAAAADx7f79+zp48KDGjRunLl262Ls5z0TgFgAAAAAAAMBbr3PnzgoKClKZMmUcfpkEyUGXSgAAAAAAAACA+DRjxoznehCao2DGLQAAAAAAAAA4GAK3AAAAAAAAAOBgCNwCAAAAAAAAgIMhcAsAAAAAAAAADobALQAAAAAAAAA4GAK3AAAAAAAAAOzKZDJp2bJl8Z73TeZs7wYAAAAAAAAAcBwfd+ugBUvmSpJcXFyULm16NajTWJ906iZn59cTTrx06ZK8vb3jPe+bjMAtAAAAAAAAkIByjyueoPU9qLf5hY8pVzpEo4dPUERkhNauX60v+nSXs7OzPunUzSZfZGSkXF1dX7mNvr6+ryXvm4ylEgAAAAAAAADYcHV1U+rUPvJLn0Etm7VRqZJl9Mefv+vjbh3Uom0TfTduuPIWya7i5YIkSRcu/qO2nVooa54Myp4vo5q3aaxz58/alDl34U8qVaGo/LKlUp7C2dSrT3frvn8vfxAZGanOnTsrTZo0cnd3V8aMGTV48OA480rSgQMHVK5cOSVKlEgpUqRQu3btdP/+fev+li1bqlatWhoxYoTSpEmjFClSqFOnToqKinoNr1z8IXALAAAAAAAA4Knc3RIpKipSkrRxy186dfqkFs5eptnTFioqKkqNmtdR4sRJtHzR7/p1yWolTpxYjVvUVWTko2Nm/DRVvXp31weNW2rDqq2aNXWe/DMGxFnXmDFj9Msvv2jhwoU6duyY5syZI39//zjzhoWFqVKlSvL29tbOnTu1aNEi/fnnn+rcubNNvvXr1+vUqVNav369Zs6cqRkzZmjGjBnx9vq8DiyVAAAAAAAAACBOhmHof5s3aMP/1urDlu1048YNeSTy0KghY61LJCxeukAWi0XfDR0nk8kkSRo9fIKy5c2gLds2qkyp8vpu3Ah91Laz2rXuYC27QL6gOOs8d+6csmbNqpIlS8pkMiljxoxPbN/cuXP18OFDzZo1S4kTJ5YkjRs3TtWrV9fQoUPl4+MjSfL29ta4cePk5OSkHDlyqGrVqlq7dq3atm0bL6/T68CMWwAAAAAAAAA21qxbpUy50ipD9tRq0rKealaro+6f9pIk5cyRy2Zd20NHDujM2dMKyJ1OmXKlVaZcaZU9v78eRjxU6Nkzunb9mi5fuaTg4qWfq+6WLVtq3759yp49uz7++GOtXr36iXmPHDmifPnyWYO2klSiRAlZLBYdO3bMmpY7d245OTlZt9OkSaOrV68+9+thD8y4BQAAAAAAAGCjRLFgDftmlFxcXOXrk0bOzv8XRvRIlNgmb1hYmPIG5tfE0VNilZMieUqZzS82d7RgwYI6c+aMfv/9d/35559q0KCBQkJCtHjx4pfrjCQXFxebbZPJJIvF8tLlJQQCtwAAAAAAAABseCRKrEz+mZ8rb57AfFq+4melTJFKSZMmizOPX/oM2rjlL5UsXuq5ykyWLJkaNmyohg0bql69eqpcubJu3ryp5MmT2+TLmTOnZsyYobCwMOus282bN8tsNit79uzPVZejYqkEAAAAAAAAAC+tbq0GSp48hZq3baJtO7bo7PlQbd66UV/2+1wXL12QJPX4tJcmTRmnKdMn6fSZU9p/cJ+mzvghzvJGjRqlefPm6ejRozp+/LgWLVokX19feXl5xcrbtGlTubu7q0WLFjp48KDWr1+vLl26qFmzZtb1bd9UzLgFAAAAAAAA8NI8Enlo+YLfNXBIX7X+6APdv39fvr5pFFyitJImSSpJaliviSIiHuqHHyeo/6Cvldw7hapVqRlneUmTJtWwYcN04sQJOTk5qXDhwvrtt9/iXHLBw8NDf/zxhz755BMVLlxYHh4eqlu3rkaNGvVa+5wQCNwCAAAAAAAACehQ5y1xpgdcNhK4JXEbM3LiC+9LndpHY0dNemq5zZu2VvOmrePcZxj/1/e2bduqbdu2Tyzn33klKU+ePFq3bt0T88+YMSNW2vfff//UtjoClkoAAAAAAAAAAAdD4BYAAAAAAAAAHAyBWwAAAAAAAABwMARuAQAAAAAAAMDBELgFAAAAAAAAAAdD4BYAAAAAAAAAHAyBWwAAAAAAAABwMARuAQAAAAAAAMDBELgFAAAAAAAAAAdD4BYAAAAAAACAQzGZTFq2bJkkKTQ0VCaTSfv27bNrmxIagVsAAAAAAAAAVh936yAff0/5+HsqXZYUKlQyjwYM7q2HDx/au2nvFGd7NwAAAAAAAAB4lzRa2ShB69sRNO+FjylXOkSjh09QVHSU/j6wTx937yCTTOrda8BraCHiwoxbAAAAAAAAADZcXd2UOrWP0qVNryqVqqlUidL6a9N6SZLFYtHo8SNVqGQeZczuo7KVS+jX35bZHH/0+BE1bd1AmQPTKyB3OtWoX1mhZ09Lkvb+vVv1P6ipnAUyKUseP5UuXVp79uxJ6C46PGbcAgAAAAAAAHiiI8cOa+fuHUqfzk+SNHrCSC1ZulDDv/1OmTJl1rbtW9Tp03ZKkTylir9XUpcuX1StBu+r+HsltWTuL0qaJJl27Nqm6OgYSdL9sPtqWLeJBvUbLkOGZsz7QVWqVNGJEyeUNGlSe3bVoRC4BQAAAAAAAGBjzbpVypQrrWKioxURGSGz2azBA4YrIiJCo8eP0qLZy1U4qIgkyT9DJm3ftVWz5k5X8fdKatqsKUqaNJl+GDtdLi4ukqTMAVmsZQcXL21T1+TJk+Xl5aW//vpL1apVS7hOOjgCtwAAAAAAAABslCgWrGHfjFJ4eLh++HGCnJydVO39mjp6/IgePAhXg2a1bPJHRUUqMFdeSdKhwwf0XuHi1qDtf129dlVDRg7Ulm2bdP3GdVksMQoPD9e5c+ded7feKARuAQAAAAAAANjwSJRYmfwzS5K+Hz5eZd8voTkLZilHtlySpDnTFiqNbxqbY1xd3SRJ7u7uTy37424f6dbtm/qm71ClT+entJlSqFixYoqMjHwNPXlzEbgFAAAAAAAA8ERms1mfdOymvt98qa3rd8vN1U0XLv6j4u+VjDN/rhyBWrBkrqKiouKcdbtj93YNHThSIWUrSpIizHd0/fr119qHN5HZ3g0AAAAAAAAA4NhqVK0lJycnzZo7XR3adVGfgb20YPFchZ49rf0H92nqjB+0YPFcSVLrFu10//49te/SSvv279HpM6e06Of5OnnqhCQpwD9Ai5bO1/GTx7R77y41bdpUiRIlsmf3HBIzbgEAAAAAAAA8lbOzs1o3b6txP4zWzo37lSJ5So2ZMEpnz4cqWTJP5c2dT5906iZJSu6dXIvn/qoBg3qrVsOqcnJyUmCuPCpcqKgk6buh49T9y09UoWoppU2bTkOHDVH37t3t2T2HROAWAAAAAAAASEDzq86PMz3gspHALYnbmJET40z/uGNXfdyxqySpXesOate6wxPLyJ0zUAt+WhrnvjyB+fTHLxus26kzJlO9evVs8hjG/70W/v7+NtvvCpZKAAAAAAAAAAAHQ+AWAAAAAAAAABwMgVsAAAAAAAAAcDAEbgEAAAAAAADAwRC4BQAAAAAAAAAHQ+AWAAAAAAAAABwMgVsAAAAAAAAAcDAEbgEAAAAAAADAwRC4BQAAAAAAAAAHQ+AWAAAAAAAAABwMgVsAAAAAAAAAVh936yAff89YP2dCT2nr9s364MOGylsku3z8PfXbHyvs3dy3lrO9GwAAAAAAAAC8S8wl68WZHvqa6vNfvOiFjylXOkSjh0+wSUuRIqVOnzmt3DkD1aT+B2r10Qfx1UTEgcAtAAAAAAAAABuurm5KndonVnr5shVUvmwFO7To3cNSCQAAAAAAAADgYJhxCwAAAAAAAMDGmnWrlClXWut2+TIhmjphlh1b9O4hcAsAAAAAAADARoliwRr2zSjrtodHYju25t1E4BYAAAAAAACADY9EiZXJP7O9m/FOY41bAAAAAAAAAHAwzLgFAAAAAAAA8FzCwu7rTOhp6/a582d18NB+eXl5K306Pzu27O1D4BYAAAAAAADAc9m3f6/qNK5m3e77zZeSpIZ1m2jMyIn2atZbicAtAAAAAAAAkIAsmxbHmR5w2UjglsTtaQHYEsWCdSX0TgK25t3FGrcAAAAAAAAA4GAI3AIAAAAAAACAgyFwCwAAAAAAAAAOhsAtAAAAAAAAADgYArcAAAAAAAAA4GAI3AIAAAAAAACAgyFwCwAAAAAAAAAOhsAtAAAAAAAAADgYArcAAAAAAAAA4GAI3AIAAAAAAACAgyFwCwAAAAAAAMDq424d5OPvKR9/T6XLkkKFSubRgMG99fDhQ5t8q9euUq0GVRSQO538c/iqUo0ymr9oTpxlrvh9uWo3rKosefyUKVdalalcXCNHD9Wt2zef2Z558+bJyclJnTp1irVvxowZ8vLyivM4k8mkZcuW2aQtWbJEZcqUkaenp5IkSaK8efNqwIABunnz2e1IaM72bgAAAAAAAADwLtnw9ZW4019Tfa07p37hY8qVDtHo4RMUFR2lvw/s08fdO8gkk3r3GiBJmjrjB/Ue8IU6f/Sphn47Sq4uLlq15jd9/tVnOnr8sPp99a21rEHDB2jcpO/V/sOO+vLzPvJJ7aszoac0c840Lfp5gb7O1/Opbfnxxx/1+eef64cfftDIkSPl7u7+wv2RpK+++kpDhw7VZ599pkGDBilt2rQ6ceKEJk2apJ9++kmffPLJS5X7uhC4BQAAAAAAAGDD1dVNqVP7SJLSpU2vxUvn669N69Vb0oWL/6jft1+pXeuO+urzvtZjOrTtIhcXV33V73NVr1JbQQUKac++3Ro9fqQG9hmidq07WPNm8Muo0sHldOfO7ae248yZM9qyZYuWLFmi9evX6+eff1aTJk1euD87duzQoEGD9P3339sEaP39/VWhQgXdvv30dtgDSyUAAAAAAAAAeKIjxw5r5+4dcnFxlST9+ttyRUVFqWO7LrHyNm/SSokTJ9HSXxZLkpYsW6jEiZOoVbM2cZbt6en11LqnT5+uqlWrytPTUx988IF+/PHHl+rDnDlzlCRJEnXs2DHO/U9absGeCNwCAAAAAAAAsLFm3SplypVWGbKlVplKxXT9xjV1av+xJOn0mZNKltRTPql9Yx3n6uqqjH7+On3mpCTpTOgpZfTzl4uLywu3wWKxaMaMGfrggw8kSY0aNdKmTZt05syZFy7rxIkTCggIeKl22AuBWwAAAAAAAAA2ShQL1rrfNur3ZWvVsG4TNarfVNXer/nC5RiG8cw8586dU5IkSaw/gwYNkiStWbNGYWFhqlKliiQpZcqUqlChgqZNm/Za2uFoWOMWAAAAAAAAgA2PRImVyT+zJOn74eNV9v0SmrNglpo2bK6ATFl0994dXb5ySb4+aWyOi4yMVOi5MypRLFiSFJApi7bv2qaoqKgnznZNmzat9u3bZ91Onjy5pEcPJbt586YSJUpk3WexWLR//371799fZrNZyZIlU1hYmCwWi8zm/5uj+njNWk9PT0lStmzZtGnTpqe2w9Ew4xYAAAAAAADAE5nNZn3SsZuGjPhGDx4+ULX3a8jFxUUTp4yLlXfmnGkKDw9T7Rr1JEl1atZXWNh9Tf9papxl37lzW87OzsqSJYv1J3ny5Lpx44aWL1+u+fPna9++fdafvXv36tatW1q9erUkKXv27IqOjrYJ/ErSnj17JD0K2EpSkyZNdP/+fU2YMCHOdjjiw8mYcQsAAAAAAADgqWpUraUBg3tr+qwp6tjuY/X+YoD6ffuV3NzcVL92I7m4OGvVmt80aNgAdWjbWUEFCkmSggoUUuf2n6jft1/p8pWLer9idfn6+OrM2dOaNWeaihQqpq/z9oxV308//aQUKVKoQYMGMplMNvuqVKmiH3/8UZUrV1bu3LlVsWJFtW7dWiNHjlRAQICOHTumTz/9VA0bNlS6dOkkSUWLFtXnn3+ubt266cKFC6pdu7bSpk2rkydPatKkSSpZsqQ++eST1/9CvgACtwAAAAAAAACeytnZWa2bt9W4H0arxQcfqv2HHZUxg78mThmrKdMnyRITo+zZcmjoN6PUuMEHNsf27jVAefPk1/RZUzVzznRZLBb5Z8ik6lVqqmHdxnHWN23aNNWuXTtW0FaS6tatq2bNmun69etKmTKlFixYoL59+6p9+/a6ePGi0qdPr9q1a6t37942xw0dOlRBQUEaP368Jk2aJIvFosyZM6tevXpq0aJF/L1Y8YTALQAAAAAAAJCAynzjE2d6wGXHeIDWmJET40z/uGNXfdyxq3W7coUqqlyhynOVWbNaHdWsVue527B///4n7mvQoIEaNGhg3fby8tLo0aM1evToZ5b732MdGWvcAgAAAAAAAICDIXALAAAAAAAAAA6GwC0AAAAAAAAAOBgCtwAAAAAAAADgYAjcAgAAAAAAAICDIXALAAAAAAAAPINFlkf/MezbDjg+w4ifXxICtwAAAAAAAMAzhMeEK9qIlhFD5BZPFx4eLklycXF5pXKc46MxAAAAAAAAwNvsfsx9Hbl7RJ6JPJXYKbFkiv86Iiz2CwpHRUfare6HDx/are74ZBiGwsPDdfXqVXl5ecnJyemVyiNwCwAAAAAAADyDIUNLriyRXyI/eT70lOk1RG6Nu/Fe5HN7eDfGbnXfjXS3W92vg5eXl3x9fV+5HAK3AAAAAAAAwHO4HX1bA08NVAqXFHLSq82mjMt3k6PjvczndaBIH7vV3bR/TrvVHd9cXFxeeabtYw4ZuB0/fryGDx+uy5cvK1++fBo7dqyKFCnyxPyLFi1S7969FRoaqqxZs2ro0KGqUqVKArYYAAAAAAAA74IYI0ZXI6++lrLNl+wXuH14x2K3ut3d364Zt/HF4R5OtmDBAnXt2lV9+/bVnj17lC9fPlWqVElXr8Y9ILZs2aLGjRvrww8/1N69e1WrVi3VqlVLBw8eTOCWAwAAAAAAAED8cLjA7ahRo9S2bVu1atVKuXLl0qRJk+Th4aFp06bFmX/06NGqXLmyevTooZw5c2rgwIEqWLCgxo0bl8AtBwAAAAAAAID44VCB28jISO3evVshISHWNLPZrJCQEG3dujXOY7Zu3WqTX5IqVar0xPwAAAAAAAAA4Ogcao3b69evKyYmRj4+PjbpPj4+Onr0aJzHXL58Oc78ly9fjjN/RESEIiIirNt37tyRJN2+fVsWi/3W8khQEWF2q/q2Kf6fuPi8jAeG3eqWpHuG/ep/EHXfbnXfvn3bbnW/FoyfBGfPsSMxfuIV4yfBvaufPRLjJz4xfhIe4yce2XHsSIwfe+DcLR4xfuyC8fP2u3v3riTJeI732qECtwlh8ODB6t+/f6z0jBkz2qE17x5vu9Z+x661F7Vn5cdr2q3qHnGvcoKX8K6OH7uOHYnx85Zg/NiBHceOxPiJT4wfO2D8vDUYP3bAudtbg/FjB4yfBHXv3j15eno+NY9DBW5TpkwpJycnXblyxSb9ypUr8vX1jfMYX1/fF8rfq1cvde3a1bptsVh08+ZNpUiRQiY7Xs2BY7t79678/Px0/vx5JUuWzN7NAd4ojB/g5TF+gJfH+AFeDmMHeHmMHzwPwzB07949pU2b9pl5HSpw6+rqqqCgIK1du1a1atWS9CiwunbtWnXu3DnOY4oVK6a1a9fq008/taatWbNGxYoVizO/m5ub3NzcbNK8vLzio/l4ByRLlow/vsBLYvwAL4/xA7w8xg/wchg7wMtj/OBZnjXT9jGHCtxKUteuXdWiRQsVKlRIRYoU0ffff6+wsDC1atVKktS8eXOlS5dOgwcPliR98sknKl26tEaOHKmqVatq/vz52rVrlyZPnmzPbgAAAAAAAADAS3O4wG3Dhg117do19enTR5cvX1b+/Pm1atUq6wPIzp07J7PZbM1fvHhxzZ07V19//bW+/PJLZc2aVcuWLVNgYKC9ugAAAAAAAAAAr8ThAreS1Llz5ycujbBhw4ZYafXr11f9+vVfc6vwLnNzc1Pfvn1jLbMB4NkYP8DLY/wAL4/xA7wcxg7w8hg/iG8mwzAMezcCAAAAAAAAAPB/zM/OAgAAAAAAAABISARuAQAAAAAAAMDBELgFAAAAAAAAAAdD4BYAAAAA7Mhisdi7CQAAwAERuAXsyGKx6PDhw/ZuBmBX0dHRmjFjhjp06CCJL6/Ai7p69ao6duyonj17SmIMAW8is9ms6OhoxcTEMIbh8K5cuaJhw4Zp7dq19m4KgKewWCyKjo62dzPwigjcAnbUoEEDffHFF9btI0eO2LE1gP2cOXNGP//8s+7evSuzmY8m4GkMw7D+32KxaM6cOZo0aZKmTZum+/fvM4aAN9CSJUtUpkwZHTp0SGazWTExMXzZhkMxDEOTJ09Wvnz5lC5dOi1cuFApUqSwd7MAPIXZbJazs7MkKTQ01L6NwUvjzB5IIIZhKCYmRoZhWGdSVKxYUadPn1ZwcLDMZrN69+5t84UceNusW7dOlSpV0qRJk6xfSJ2dnVWuXDmlSJFCixYtksSMQSAuhmFow4YNMplM1m2z2azTp0/rvffeU9q0aTVv3jxJjCHA0T0+L3w8VgMCAmQ2mzV69GhVrlxZAQEBOnXqlJ1bCTxy8OBBZcmSRd26dVObNm105swZ7dq1S/nz57d30wA8xcmTJ9WxY0dlypRJTZo00aVLl+zdJLwEArdAArBYLDKZTHJycpLJZNK9e/cUFRWlZcuW6fDhwwoPD9fOnTu1ePFi6xdy4G1x+vRpPXz4UJL0888/a82aNerUqZMGDBigu3fvSpKyZMmiggULav78+ZLEOAD+JTo6WsePH1dAQIDKlSunqVOnKioqyjpOUqRIoejoaJUvX94auGUMAY7p8QX8x+eFj2fIHz16VNu2bdPs2bPl5+enX375RdmzZ7dza/GuunLliiIiIqzbiRMnVkBAgGrVqqUuXbooXbp0dmwdgMcsFotiYmLi3Hf8+HF98MEHOnPmjMaNG6e+fftyYf8NReAWSABms1l3797VsGHDVKxYMTVt2lQmk0lTpkxR06ZN5e/vr9SpU0tilhTeLoMGDVLNmjW1c+dOSVK9evVUpEgRVa1aVT///LNat24tSUqXLp1KlSqlw4cP68yZMwSdAEnnz59XwYIFNXnyZKVJk0aFCxeWyWTS2rVrNWbMGElSRESEzGazihcvrmzZsunEiROMIcCBmUwmmc1m3bhxQyNGjFDdunV19uxZ5cyZU126dFH+/PnVrFkz5cuX74lfxoHX4dq1a+rTp4+yZ8+u8uXLq2XLltY1bNOlS6eqVatq1apVkh59t9m7d6+aN2+uXbt2SRJ3DQJ2YDab5eTkJEnauHGjrly5Yt23YMEC3bp1S7///ruqVq2qSpUqcdHlDUXgFognhmHYrEX275OX/fv3q1y5cpo/f76aNm2q+vXr69y5c9aToNDQUG3atMkezQbizfXr160P24uKipIkVahQQRaLxbp+c3BwsJIkSaL06dNr+PDh2rNnj5o2baqzZ8+qfPnySpkypWbPni2JixjAnTt3dPXqVaVLl05JkiRRcHCwPD09Va1aNY0aNUpbtmyRm5ubDh06pFSpUql06dJKnjy55syZI4kxBNjTk2ZBRUVFaeDAgfL399fy5cuVNm1anTx5Uvnz51fbtm2VJEkSLVu2TJJYrxoJ4tKlSypQoIDSpUuntWvXqnv37hoxYoROnz6tIUOG6ObNm3J1ddV7770nd3d3Va1aVbly5VKlSpV08+ZNubu7S+JODyAh/PcCydWrV9W2bVslS5ZMbdu2Va1atfTll19KkjJlyqRz585p5syZ+vbbbzV27FiNGzdOx44ds0fT8Qo4GwDiiclksi78feXKFZuTl4ULF8rJyUl79uxR586d1aJFCwUEBEiSKleurESJEmn79u2Kjo7mJB1vpHXr1il16tQqXbq0/vjjD7m4uEiSChcuLF9fX23btk2XL1+Wk5OTSpQooX379ikgIEA///yzLly4oCZNmujWrVuqWLGili5dKokvrHh3rFixQgcPHpRke0KePn16XblyRd7e3jKZTAoKClKiRImULFkyffzxx+rTp4+2bNmi/Pnz68iRI8qZM6eKFy+u5cuXS2IMAfb071lQkZGR1vQDBw5owYIFmjt3rjZu3KixY8eqfPnykqQcOXIoe/bs2r9/vy5duiSTycQFGLwWt27dsv4/TZo0+vvvv9W/f39t3rxZbdu2VeXKldWyZUtdvXpVFy5ckPRoHeZKlSrpzz//1DfffKP9+/drxYoVCgwMtFc3gHeOyWTStWvXJD26QDh37lydPXtWq1at0p49e/TNN99o5MiRWrlypRo3bqwWLVpoxIgROnLkiFavXq1BgwapTZs22r17t517ghfBGT0QT27evKkePXooffr0ql69ujp27KiTJ09KklxcXHTx4kXt2bNHM2bM0NKlS/Xnn3/qxo0b8vLyUqFChXT06FEdOHDAzr0Ans8///xjs50rVy6lSJFCbm5u6tChg+bNm2edgV65cmUdOHBAhw4dkiTVqFFDDx480K+//qr8+fNr0aJFcnJyUps2beTi4qKbN29qz549krjtDm+vjRs3auLEidqwYYM++OADdezYUTdv3rS56Hfx4kXrbAlJypw5s4oXL64pU6aoZ8+eCg4O1gcffKCTJ08qW7ZskqTSpUvr8uXL3LoKJKB/P2Ts8b+nT59W586dVaRIEXXq1Ml6UdIwDB0+fFj58uXT8ePHtX37dl2/fl03btyQJJUoUUJhYWH666+/JHEBBvHnwIEDateunQIDA9WwYUNNmDDB+qCimjVrat26ddY7piTp0KFDKlSokPLkySNJSpkypYKDg+Xu7q5y5crJ19eXzxjgNYnrol1MTIzGjx+v4sWLS3r0+TBs2DB9/fXXKl68uM6cOaNt27YpKirKuszJpEmTdODAAY0dO1a//vqrtm3bpu3bt+vOnTsJ2h+8Gs4EgOdksVhslkL4t5iYGA0bNkwbNmzQqFGj1KVLF23YsEEtWrTQ/fv31aFDB2XKlEnvv/++Fi9erKFDh6pmzZpq1aqVoqOj1aBBA12+fFnjxo3T4sWLVbt2bV2/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JGdnZ9WvX18rV660yde/f381a9ZMLi4uGjBggPr37y8/Pz+tW7fOJt/JkydVr149VahQQSNHjpS3t7datmz5xCDcv926dUuVK1dWvnz5NHLkSOXIkUM9e/bU77//bs1jsVhUo0YNjRgxQtWrV9fYsWNVq1Ytfffdd2rYsOGLvFxPtG7dOpUqVUp3795V3759NWjQIN2+fVvlypXTjh07rPl27typLVu2qFGjRhozZow++ugjrV27VmXKlFF4eHiscjt27KjDhw+rT58++uKLL16o308SExOjSpUqKUWKFBoxYoRKly6tkSNHavLkydY8FotF1atX17x589SiRQt9++23unTpklq0aPFCr4uHh4dq1qxp8/v1999/69ChQ0+8MNCmTRv16dNHBQsW1HfffafSpUtr8ODBsX4f+/Tpo969eytfvnwaPny4AgICVLFixVgzk8PDw1W6dGnNnj1bzZs315gxY1SiRAn16tVLXbt2faH+AAAAvJEMAAAAvDGmT59uSDJ27txpjBs3zkiaNKkRHh5uGIZh1K9f3yhbtqxhGIaRMWNGo2rVqjbHPs73WGRkpBEYGGiUK1fOmnbixAnDbDYbtWvXNmJiYmzyWywW6/8zZsxoSDL+97//WdOuXr1quLm5Gd26dbOmrV+/3pBkrF+/3ppWunRpQ5Ixa9Ysa1pERITh6+tr1K1b15r2008/GWaz2di4caNNOyZNmmRIMjZv3vzU16pFixZG4sSJn7jfYrEYWbNmNSpVqmTTt/DwcCNTpkxGhQoVbNL+a+vWrbH68fj9KVmypBEdHW2T/3n7febMGUOSMX36dJu+SDIGDBhgU2aBAgWMoKAg6/aSJUsMScb3339vTYuJiTHKlSsXq8y4PH6/Fi1aZKxYscIwmUzGuXPnDMMwjB49ehgBAQHWvuTOndt63L59+wxJRps2bWzK6969uyHJWLdunWEYj35HXF1djapVq9q85l9++aUhyWjRooU1beDAgUbixImN48eP25T5xRdfGE5OTtZ2GYZhSDL69u371L4BAAC8aZhxCwAA8IZq0KCBHjx4oBUrVujevXtasWLFE2dDSlKiRIms/79165bu3Lmj4OBg7dmzx5q+bNkyWSwW9enTJ9a6tCaTyWY7V65cCg4Otm6nSpVK2bNn1+nTp5/Z9iRJkuiDDz6wbru6uqpIkSI2xy5a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\n"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "confusion_matrix_all_six_models",
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: XGBoost ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "XGBoost Confusion Matrix: [[275, 296], [88, 1350]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: Random Forest ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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fv26x3Pp+tWrV4/22n379jWPrVixItpjY8aMMY/pT6emTZuaZVu3bo22/pkzZzzu58+f39zc9e/f3/x+27ZtHZGRka7lv/32m8Pf39+UPzw8PFr5QkJCHLt373Ytv3LliqNo0aKOVKlSOf766y9HXGhZ0qVLZ/7ftWtXR5o0aRwnTpxwPf7QQw85SpQoYf5fr14987oHDx50PX7t2jXHsWPHoj3vjh07HIGBgY7atWvf9vNzp38Pfbx06dKOs2fPxvlvo5+9vo9ChQo5Ll686Dh69KgjS5YsjqxZs8b5s0gqzs9g8ODBHsv1+/zEE0+Yxz766COPxy5cuOD181i+fLn5e7/44osey+P7HdHPU9cvVaqU49KlS67l+rfNli2beUy3Mfey6metyxcuXOjx2m+88YZZ/sILL0T7runy5s2bO27evOlaPmTIELNcv+c9e/b02AY6d+5sHps+fbojLpz7gY4dO5rPwP3m/nmvWrXKrFe8eHHz2TqdO3fOfD762OrVq6N9PnobPXp0tNf95ptvzGO637px44Zrue5zGjVqZB779ddfzTJ9PT8/P8cjjzziiIiI8HgevX/+/Pk4by8pAWF+jyhWrJj5MrrvUG5HN4D06dObHfHly5ejPV6nTp1oG5tzI/a2oTkf+/33330a5nv27Lnte/MW5vfff78jbdq0JoCi6tSpk3nucePGRStfnz59Yiz7nDlzbluWqGGuBw/6ux9++KG5v379enN/xIgRMYZ5bHQnpgcj7ju3uIb57NmzvT4e299m5MiR5rF27dq5DiJjep7k5PwMatWq5Qqabt26mWDR5VWqVPEI1NvRg60CBQp4LIvvd8R58OwtNN9///1oYa7bni7Tg4+o9GBKD6R0G3Y/gHaG+eHDhz3WP3LkiFmuB39Rt3fn63h7H944t3VvNz2wcdIDDV02efLkaM8xYcKEaAcjzjAvW7as19ctWbKkI2PGjOZgKSrd7+jvvv766+Z+WFiYuV+1alWPAxdvbAhzmtkRI20+1GY4bcrSfqeodLk2X+mccG3edffII49EW985qlWbrHxBuwBmzJhhBtxo85o29Wq54jJQS+e6a7N08eLFvY6+1ff67bffmveqU/h8+V7LlCljmvK0qf2tt94yTa7aVKhNnrHRsum4CG121WlV2izvTpt0tek0PipUqHBHzabaLaED+ZT2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bHhr+dEw15Hl2u00depUM9ajdOnSprlc+9iddCS7HsTqNLPYuoN0LIg2pXv77muw6+wNDe66deuaaaJ6Eiad3qYzTeA7XDUNAADLcagEAIDlCHMAACxHmAMAYDnCHAAAyxHmAABYjjAHAMByhDkAAJYjzAEAsBxhDligX79+5qxzzpueoUvP4KWni/UlPXuXnubTaezYseb1YzvdZ1SzZs2SL774wqfl8uZOyurN888/b84omBgSq0xAVIQ5YAm96pVe3EJvej55vbSrXtJyx44dSVaGhg0bmtfX03MmZ5gD8MQlUAFL6Lmt9WpwTnpuea2d6sVk9BKXUenlL/XSr3oJzMSiLQJcxhVIeaiZA5bSC7losOpV2dybg3/66Sdz0RkN8blz55rHtDatzfJ6GUq9GEebNm3k1KlTHs+nl4t96qmnzIVi7rvvPhk6dGicmomvX78u7777rtx///3mNfPkyWPK4izT999/by6L6ewicD6WmOW6U3rJ0PLly5vX1kvWPvnkk7J3716v6y5YsMB8vunTpzfXq1+/fr3Xz0cvNarraFnfeecdrhaGJEHNHLBUeHi4aWp3vwSlBp9ealXDVcNebxqYNWrUkAYNGpjrUF++fNk8rpdo1cec9L5eolWb8LUZ/cMPP5SjR4+6rigXE70ql17T/b///a9pOTh9+rTMmDHDPPbee++Z+3qpWb2KnXLW7H1drrjQ5+3atavkz5/ffJ7aylGlShUT6HqVMacTJ06Yy+bq2IXMmTObMtSrV8913XqlV+x78803pWfPnuYgQS8z6gxzXR/wKQeAFK9v376OjBkzOm7evGluBw8edDRt2tShm/DChQvNOs8995y5v379eo/frVatmqNKlSqOyMhI17KdO3c6/Pz8HPPnzzf3FyxYYH532bJlrnUuXLjgCAoKcuTPn9+1bMyYMWa906dPm/uLFy829ydOnBhj2bVcDz30ULTliVkub6KW9XYiIiIcV65ccQQGBjq+/vprj/LHVIa3337b3A8PDze/17t3b4/n/PLLLx0BAQGOM2fO3FGZgLiimR2whNZc06ZNa24FCxaUFStWmL5yrSE6Zc2aVSpWrOi6f+XKFXN96ebNm5saYkREhLnpdd3z5s0rmzZtMutt2LDBNDVrk7eT3tdrZMdm2bJlpvm7VatW8Xovvi5XXGlTeZ06dcznpjV9fS+XLl2K1tQeUxm0fOqXX34xv6fvx/le9KbrXL16NUkHKeLeRDM7YNFo9tWrV5t+52zZspnQ00Fx7nLmzOlx//z58yYstelXb1Fpc7WzGdnbwLaozxeVNvOHhoaaMsWHr8sVF0eOHJG6detKuXLl5OuvvzbdFf7+/mbE/rVr1zzWjakM2pSunGMIypYt6/W1nO8H8BXCHLCEBrcGT2yihqr2Mesy7c9u0qRJtPX1oEBpIGvfdlQnT56M9fW0RquBqyPn4xPovi5XXCxcuNDUprV/3znVTmvT586di7ZuTGXQ8iln/7o+lx5kRaUtKYAvEebAXUxHiVeuXNnUID/44IMY19NpbmFhYWYgm7M5We8vXbrUYyBYVNqMPGTIEJkyZYq0bNnS6zpa241a0/V1ueJCm7/1gEK7LZz0fWigRxVTGbp06WLu63vRJnodUPf0008nqFzAnSDMgbvcRx99ZEJIw1b7tnU0tobOkiVLpEOHDmZEef369U0Tcdu2bU04a0118ODBEhwcHOtza5jraPQXXnhB9u/fb/rrtWY7bdo0M0JdFS9eXEaPHi2TJk2SIkWKmFq3zo/3Zbnc6fS8oKAgj2U6xcwZzPpaL7/8spk+p6PQvZ0QRw8cOnbsKP3793eNqNfWCD0TndJlAwYMMKPZ9T1o2VOnTi0HDhyQ2bNny/Tp003YAz4T56FyAJJ9NHtsYho1rjZt2uRo0KCBIyQkxIyuLlKkiOOVV15xHD161LWO/r9hw4aO9OnTO0JDQx2DBg1yvPrqq7GOZldXr141o7rz5cvnSJs2rSNPnjyOF154wfV4WFiYo1WrVo6sWbOa39VyJna5vHGW1dvt/fffN+uMGzfOcf/995vnrlSpkmPjxo3mebt06RLtc503b56jePHiDn9/f0eZMmUca9eujfaakyZNcpQvX968l+DgYLPee++9Z2YgxPT5AYnBT//x3aECAADwNaamAQBgOcIcAADLEeYAAFiOMAcAwHKEOQAAliPMAQCwHGEOAIDlCHMAACxHmAMAYDnCHAAAyxHmAABYjjAHAEDs9n+x8Hru2uExUwAAAABJRU5ErkJggg=="
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Random Forest Confusion Matrix: [[134, 437], [14, 1424]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: SVM ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "SVM Confusion Matrix: [[271, 300], [74, 1364]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: Naive Bayes ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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d98wHzBggHl9mzZtnJCQEPfjv/32m5MiRQqz/MHBweGWLyAgwNmzZ49HqBUuXNiE64kTJ5zohLl69dVXnWTJknnsbB999FETHspbmOuO8fjx4+Hmu3PnTsfX19epVavWfT8/bzt8DZXz589H+bvRz17XQw8AL1++7Bw7dszJkCGD2XlH9bNIiDDX71vDQL+z7du33zfMDxw4EG5+ur76Hen2cPXq1Ui3cX0/PcDVQAodamrHjh1m+ueff95j+/X39zcHVocPHw4XtDq9bjdR4fpuNZBdB3G9evVyateubda/Zs2aHgcR6vTp02b9wpo0aZKZ14cffujxeLFixcxv/cqVKx6PaxDr9D169HjgdWsajd+5rahmR4T+/vtv8zdnzpxR/pS06larwrRqXqtaQ6tXr548/fTTpvreWxW0dqDT6jIXbV9t3LixXL58Wfbu3Run35S3amhdh/uZNGmSqUp29WB2KV26tLzwwgty6dIl0zkoLK1SL1KkiMf7axWsNinoyIEHqWrXDm66POqXX36RXbt2mccjolWN2tM9rEcffVSqV69uqnNv374d7WXRppHodArT9nytStc26K5du5oREhcuXDDt/Volmli5mgD0O9MRFfeTP3/+cI9pZzKtsg8KCpItW7bc9/3atGkjFy9eNB1PQ/vuu+/c1dUu2n6tQzi1171WP4emVdNlypQxndeiQ/sH6Pert48//liWLVtmOjTqtpsuXTqPabVq3dVZLjT9fv39/WXFihUej2tnQv2th12mr776yvzt0qVLjNct9QP+zm1ABzjEqj179pg2Uw0Dbz1x9XEdyqa9YLW9KrTHH3883PSuAwkNxbigP/zZs2ebdj0dYlOzZk2zXFHpqKU7E+3RX7RoUa8HPLqu2n6p66o7sLhcVz140Pb3CRMmyFtvvWWCUNsQQ+/cvdFl034RP/30kzl4Cxve2g4a+gArKsqXLx/t5e/evbssXbrUdORTGuo6aiIq9PMaNWqUxJS2T0d3XLd+x9pmrKMGtA9H1apVI5xW24n1oE+nPXLkSLj+EdrGfD+6HWmAaXhr3walBxPa/0JDSQ+YXTZt2uQ+sNMDpbD0d6rfr96i2jHx1KlT7g5wuvx6YK4jWV588UX5888/w3UG1N/WF198Ib/99ps5CNGhkhGtb/v27c1Bkf5mtH+AOn36tOn3UblyZSlW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"
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Naive Bayes Confusion Matrix: [[365, 206], [372, 1066]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: RBF ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "RBF Confusion Matrix: [[114, 457], [39, 1399]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Generating Confusion Matrix for: ANN ...\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "ANN Confusion Matrix: [[345, 226], [128, 1310]]\n\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Saved confusion_matrices.json successfully!\n"
     ]
    }
   ],
   "source": [
    "# ============================================================\n",
    "# Confusion Matrices for All Six Models — Final Holdout Set\n",
    "# ============================================================\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
    "import matplotlib.pyplot as plt\n",
    "import json\n",
    "\n",
    "class_labels = [\"Inactive\", \"Active\"]\n",
    "cm_results = {}\n",
    "\n",
    "for model_name, build_model in model_builders.items():\n",
    "    print(f\"Generating Confusion Matrix for: {model_name} ...\")\n",
    "    \n",
    "    # Reuse the already-fitted best_model for XGBoost to avoid retraining\n",
    "    if model_name == best_model_name:\n",
    "        fitted = best_model\n",
    "    else:\n",
    "        fitted = build_model()\n",
    "        fitted.fit(X_dev, y_dev)\n",
    "    \n",
    "    preds = fitted.predict(X_holdout)\n",
    "    cm = confusion_matrix(y_holdout, preds)\n",
    "    cm_results[model_name] = cm.tolist()\n",
    "    \n",
    "    # Display individual confusion matrix\n",
    "    fig, ax = plt.subplots(figsize=(6, 5))\n",
    "    disp = ConfusionMatrixDisplay(\n",
    "        confusion_matrix=cm,\n",
    "        display_labels=class_labels\n",
    "    )\n",
    "    disp.plot(\n",
    "        ax=ax,\n",
    "        values_format=\"d\",\n",
    "        cmap=\"Blues\",\n",
    "        colorbar=False\n",
    "    )\n",
    "    \n",
    "    plt.title(f\"Confusion Matrix — {model_name}\", fontsize=14, pad=12)\n",
    "    plt.xlabel(\"Predicted Label\", fontsize=11)\n",
    "    plt.ylabel(\"Actual Label\", fontsize=11)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(f\"{model_name} Confusion Matrix: {cm.tolist()}\\n\")\n",
    "\n",
    "# Save confusion matrices data for backend and frontend\n",
    "with open(\"confusion_matrices.json\", \"w\") as f:\n",
    "    json.dump(cm_results, f, indent=2)\n",
    "\n",
    "print(\"Saved confusion_matrices.json successfully!\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "X-h5dQ9IJ_Rl",
   "metadata": {
    "id": "X-h5dQ9IJ_Rl"
   },
   "source": [
    "## Section 17 — Simple Prediction Function\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "98R5BkVeJ_Rl",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "98R5BkVeJ_Rl",
    "outputId": "17de0843-0e01-477a-b2c7-814724bef195"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "{'smiles': 'CC1(C)OC(=O)C(OC2CCCCC2)=C1c1ccc(S(C)(=O)=O)cc1', 'predicted_class': 'Active', 'probability_active': 0.9200300574302673}\n",
      "{'smiles': 'CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1', 'predicted_class': 'Active', 'probability_active': 0.6495847105979919}\n"
     ]
    }
   ],
   "source": [
    "def predict_molecule(smiles, model=None):\n",
    "    \"\"\"\n",
    "    Predict Active / Inactive for a single molecule given its SMILES string.\n",
    "\n",
    "    Steps:\n",
    "      1. Parse the SMILES string with RDKit.\n",
    "      2. Generate the same 2048-bit Morgan fingerprint used for training.\n",
    "      3. Run the saved/selected model on that fingerprint.\n",
    "      4. Return the predicted class and, if available, the probability.\n",
    "    \"\"\"\n",
    "    if model is None:\n",
    "        model = joblib.load(\"best_model.pkl\")\n",
    "\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    if mol is None:\n",
    "        return {\"smiles\": smiles, \"error\": \"Invalid SMILES string -- could not be parsed by RDKit.\"}\n",
    "\n",
    "    fingerprint = smiles_to_fingerprint(smiles).reshape(1, -1)\n",
    "\n",
    "    predicted_class = int(model.predict(fingerprint)[0])\n",
    "    class_label = \"Active\" if predicted_class == 1 else \"Inactive\"\n",
    "\n",
    "    result = {\"smiles\": smiles, \"predicted_class\": class_label}\n",
    "\n",
    "    if hasattr(model, \"predict_proba\"):\n",
    "        result[\"probability_active\"] = float(model.predict_proba(fingerprint)[0, 1])\n",
    "\n",
    "    return result\n",
    "\n",
    "\n",
    "# Quick test with a couple of sample SMILES strings taken from the dataset.\n",
    "sample_smiles = [\n",
    "    \"CC1(C)OC(=O)C(OC2CCCCC2)=C1c1ccc(S(C)(=O)=O)cc1\",  # known active example from the dataset\n",
    "    \"CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1\",  # known inactive example from the dataset\n",
    "]\n",
    "\n",
    "for smiles in sample_smiles:\n",
    "    print(predict_molecule(smiles, model=best_model))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "mFC_3stuJ_Rl",
   "metadata": {
    "id": "mFC_3stuJ_Rl"
   },
   "source": [
    "## Section 18 — Final Summary\n",
    "\n",
    "```\n",
    "Raw dataset (only input dataset)\n",
    "    |\n",
    "Preprocessing (drop missing Value/SMILES)\n",
    "    |\n",
    "SMILES validation (RDKit)\n",
    "    |\n",
    "Create Activity from Standard Value\n",
    "    |   Standard Value < 10,000  -> Active\n",
    "    |   Standard Value >= 10,000 -> Inactive\n",
    "    |\n",
    "Final V2 dataset (CHEMBL230_Preprocessed_Data_V2.csv)\n",
    "    |\n",
    "Morgan fingerprints (radius=2, 2048 bits)\n",
    "    |\n",
    "70/30 molecule-aware split (0 molecule overlap)\n",
    "    |\n",
    "10-fold molecule-aware, stratified CV\n",
    "    |\n",
    "Six models compared on Mean CV F1\n",
    "    |\n",
    "Best model selected automatically\n",
    "    |\n",
    "Evaluated once on the untouched holdout set\n",
    "    |\n",
    "best_model.pkl + metadata.json saved\n",
    "    |\n",
    "predict_molecule(smiles) ready to use\n",
    "```\n",
    "\n",
    "The winning model, and every metric reported above, comes directly from the experiment — no model is assumed or hard-coded to win.\n",
    "\n",
    "The important preprocessing change is that the `Activity` target is now generated directly from the raw `Standard Value`; the earlier V1 dataset is not read or used anywhere in this notebook.\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.x"
  },
  "modified_note": "Raw-only preprocessing: Activity created from Standard Value <10000 Active, >=10000 Inactive. No V1 reference dataset."
 },
 "nbformat": 4,
 "nbformat_minor": 5
}