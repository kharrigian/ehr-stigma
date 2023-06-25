
"""
Ensure that all default seaarch and models behave as expected
"""

######################
### Imports
######################

## External
import pytest
import numpy as np
import pandas as pd

## API
from stigma import settings
from stigma import StigmaSearch
from stigma import StigmaBaselineModel
from stigma import StigmaBertModel

######################
### Fixtures
######################

@pytest.fixture
def model_inputs():
    ## Examples
    documents = [   
    """
    Miss Doe is a charming, 73 year old women who visits us today with a chief complaint 
    of heart pain. Unfortunately, not a good historian.
    """
    ]
    ## Anchor Search
    search_tool = StigmaSearch()
    search_results = search_tool.search(documents)
    ## Inputs
    model_inputs = search_tool.format_for_model("other",search_results)
    return model_inputs

######################
### Tests
######################

def test_load_baseline():
    """
    
    """
    ## Model Type
    baseline_models = {x:y for x, y in settings.MODELS.items() if y["model_type"]=="baseline"}
    bert_models = {x:y for x, y in settings.MODELS.items() if y["model_type"]=="bert"}
    ## Identify Models
    assert list(baseline_models.keys()) == [
            'mimic-iv-discharge_majority_overall',
            'mimic-iv-discharge_majority_keyword',
            'mimic-iv-discharge_logistic-regression_context',
            'mimic-iv-discharge_logistic-regression_keyword-context'
    ]
    ## Load Each Model
    for model_id, model_params in baseline_models.items():
        assert len(model_params["tasks"]) == 3
        for task in model_params["tasks"].keys():
            model = StigmaBaselineModel(model=model_id,
                                        keyword_category=task,
                                        preprocessing_params=None,
                                        batch_size=None)
    ## Check Exceptions
    with pytest.raises(KeyError):
        _ = StigmaBaselineModel(model=list(baseline_models.keys())[0],
                                keyword_category="not_a_real_task",
                                preprocessing_params=None,
                                batch_size=None)
    with pytest.raises(ValueError):
        _ = StigmaBaselineModel(model=list(bert_models.keys())[0],
                                keyword_category="adamant")
        
def test_load_bert():
    """
    
    """
    ## Model Type
    baseline_models = {x:y for x, y in settings.MODELS.items() if y["model_type"]=="baseline"}
    bert_models = {x:y for x, y in settings.MODELS.items() if y["model_type"]=="bert"}
    ## Identify Models
    assert list(bert_models.keys()) == [
            'mimic-iv-discharge_base-bert',
            'mimic-iv-discharge_clinical-bert'
    ]
    ## Load Each Model
    for model_id, model_params in bert_models.items():
        assert len(model_params["tasks"]) == 3
        for task in model_params["tasks"].keys():
            model = StigmaBertModel(model=model_id,
                                    keyword_category=task)

def test_search(model_inputs):
    """
    
    """
    assert isinstance(model_inputs, tuple) and len(model_inputs) == 3
    doc_ids, keywords, text = model_inputs
    assert doc_ids == [0, 0]
    assert keywords == ["charming","historian"]
    assert text == ['miss doe is a charming, 73 year old women who visits us today with a',
                    'of heart pain. unfortunately, not a good historian.']

def test_predict_baseline(model_inputs):
    """
    
    """
    ## Initialize Models
    majority_model = StigmaBaselineModel("mimic-iv-discharge_majority_keyword",
                                         keyword_category="other")
    lr_model = StigmaBaselineModel("mimic-iv-discharge_logistic-regression_keyword-context",
                                   keyword_category="other")
    ## Make Predictions
    pred_majority = majority_model.predict(text=model_inputs[2],
                                           keywords=model_inputs[1])
    pred_lr = lr_model.predict(text=model_inputs[2],
                               keywords=model_inputs[1])
    ## Check
    assert all(isinstance(i, pd.DataFrame) for i in [pred_majority, pred_lr])
    assert pred_majority.shape == pred_lr.shape == (2, 4)
    assert pred_majority.columns.tolist() == pred_lr.columns.tolist()
    assert np.isclose(pred_majority.max(axis=1), np.array([0.80952381, 0.79069767])).all()
    assert np.isclose(pred_lr.max(axis=1), np.array([0.88565109, 0.627966])).all()
    assert (pred_majority.idxmax(axis=1) == pred_lr.idxmax(axis=1)).all()
    assert pred_majority.idxmax(axis=1).tolist() == ["positive","negative"]

def test_predict_bert(model_inputs):
    """
    
    """
    ## Initialize Model
    bert_model = StigmaBertModel("mimic-iv-discharge_clinical-bert",
                                 keyword_category="other")
    ## Make Predictions
    pred_bert = bert_model.predict(text=model_inputs[2],
                                   keywords=model_inputs[1])
    ## Check
    assert isinstance(pred_bert, pd.DataFrame)
    assert pred_bert.shape == (2, 4)
    assert np.isclose(pred_bert.max(axis=1), np.array([0.9975394 , 0.98821485])).all()
    assert pred_bert.idxmax(axis=1).tolist() == ["positive","negative"]