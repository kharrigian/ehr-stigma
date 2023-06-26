
"""
Simple test suite to confirm MIMIC-IV dataset (and annotations) loads appropriately
"""

###################
### Imports
###################

## External
import numpy as np
import pandas as pd

## Repository
from stigma import util

###################
### Functions
###################

def test_load_mimic_source():
    """
    
    """
    ## Initialize Loader
    loader = util.load_mimic_iv_discharge(clean_text=True,
                                          normalize_text=True,
                                          as_iterator=True,
                                          sample_rate_chunk=0.01,
                                          sample_rate_note=0.01,
                                          chunksize=1000,
                                          random_state=42,
                                          verbose=True)
    ## Load Subset of Notes (Chunked)
    loaded_notes = []
    for chunk in loader:
        loaded_notes.append(chunk)
    ## Concatenate Notes
    loaded_notes = pd.concat(loaded_notes).reset_index(drop=True)
    ## Validate Loading Procedure
    assert loaded_notes.shape == (50, 10)
    expected_cols = ['encounter_note_id',
                     'enterprise_mrn',
                     'encounter_id',
                     'encounter_type',
                     'note_seq',
                     'encounter_date',
                     'encounter_date_aux',
                     'note_text',
                     'encounter_note_service',
                     'encounter_note_unit']
    assert all(col in loaded_notes.columns for col in expected_cols)
    assert (loaded_notes[["encounter_note_id","encounter_note_service","encounter_note_unit"]].values[0] == np.array(['12199246-DS-11', 'CMED', 'Medicine/Cardiology'])).all()
    assert (loaded_notes[["encounter_note_id","encounter_note_service","encounter_note_unit"]].values[-1] == np.array(['18967046-DS-3', 'MED', 'Medicine'])).all()

def test_load_mimic_annotations():
    """
    
    """
    ## Load Annotations
    annotations = util.load_annotations_mimic_iv_discharge()
    ## Validate
    assert isinstance(annotations, pd.DataFrame)
    assert annotations.shape == (5043, 13)
    assert annotations["enterprise_mrn"].values[0] == 10658748
    assert annotations["enterprise_mrn"].values[-1] == 19625524
    assert annotations.groupby(["keyword_category","label"]).size().to_dict() == {
        ('adamant', 'difficult'): 526,
        ('adamant', 'disbelief'): 609,
        ('adamant', 'exclude'): 115,
        ('compliance', 'negative'): 893,
        ('compliance', 'neutral'): 439,
        ('compliance', 'positive'): 271,
        ('other', 'exclude'): 496,
        ('other', 'negative'): 1221,
        ('other', 'neutral'): 96,
        ('other', 'positive'): 377
        }

def test_load_mimic_annotations_metadata():
    """
    
    """
    ## Load Metadata
    metadata, _ = util.load_annotations_metadata_mimic_iv_discharge(annotations=util.load_annotations_mimic_iv_discharge(),
                                                                    clean_text=False,
                                                                    normalize_text=False,
                                                                    load_all=False,
                                                                    load_source=False,
                                                                    random_state=42,
                                                                    verbose=True)
    ## Validate
    assert isinstance(metadata, pd.DataFrame)
    assert metadata.shape == (4710, 15)
    assert len(metadata["enterprise_mrn"].unique()) == 4259
    assert metadata["split"].value_counts().to_dict() == {'train': 3274, 'dev': 968, 'test': 468}
    assert (metadata.groupby(["enterprise_mrn"]).agg({"split":lambda x: len(set(x))})["split"] == 1).all()
