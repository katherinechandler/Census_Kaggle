# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score


def split_fit_model(X, y, validation_size, estimator):

# Splits defined X and y arrays by validation size, fits the model (estimator) and 
# calculates classification report for assessment
# Here X is census feature data and y is income class(above 50K = 1, below 50K = 0)

    X_train, X_validation, y_train, y_validation = train_test_split(X,y,
    test_size=validation_size,random_state=42)

    clf = estimator(random_state=42)

    clf.fit(X_train, y_train)
    
    y_val_pred = clf.predict(X_validation)

    report = classification_report(y_validation, y_val_pred)
    score = precision_score(y_validation, y_val_pred)
    
    print('The classification report of {}, default params \n'.format(estimator.__name__))
    print(report)

    # return fitted classifier and precision and accuracy scores for later
    return clf, precision_score(y_validation, y_val_pred), recall_score(y_validation, y_val_pred)





def calc_fpr_trp(X, y, validation_size, estimator):
        
# calculates the false positive rate and true positive rate of estimators
# use .predict_proba or decision_function depending on estimator

    X_train, X_validation, y_train, y_validation = train_test_split(X,y,
    test_size=validation_size,random_state=42)

    clf = estimator(random_state=42)

    clf.fit(X_train, y_train)
    
    try: # for classes with a predict_probab method
        y_scores = clf.predict_proba(X_validation)[:, 1] # score = proba of positive class
        fpr, tpr, thresholds = roc_curve(y_validation, y_scores)
        
    except AttributeError: # for classes without a decision_function method (loss = hinge)
        score_roc = clf.decision_function(X_validation)
        fpr, tpr, thresholds = roc_curve(y_validation, score_roc)
        
    return(estimator.__name__,fpr, tpr)


def plot_roc_curve(fpr, tpr, label=None):
# plot roc_curve give the fpr and tpr values

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


def rank_features(X, y, estimator, feature_list):
# use RFE to rank feature importance for a given estimator

    est = estimator()
    rfe = RFE(est)
    rfe = rfe.fit(X,y)
    keys = feature_list
    values = rfe.ranking_
    dictionary = dict(zip(keys, values))
    scores = pd.DataFrame.from_dict(dictionary, orient='index')
    scores.columns = ['{}_rank'.format(estimator.__name__)]
    return(scores)
