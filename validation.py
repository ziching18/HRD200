## import functions
from modules import *

plotStyle()
sns.set_palette(sns.color_palette("Spectral"))

def main(dataset_file, training_path, truth_file=None):
    ## import dataset
    dataset = pd.read_csv(dataset_file, index_col="id")
    cohort = "tcga"

    if truth_file == None:
        no_truth = True
    else:
        no_truth = False

    if no_truth == False:
        truth = pd.read_excel(truth_file, index_col="id", engine="openpyxl")
        #dataset = dataset.loc[list(truth.index)] 
        dataset = dataset.reindex(truth.index).dropna()
        truth = truth.reindex(dataset.index)

    labels = {
        "svc": "Support Vector Machine",
        "rf": "Random Forest",
        "avg": "Composite Model"
    }

    mean_tests = []

    for sets in [1,2,3,4,5]:
        trained_models = []
        proba_dfs = []
        for random_state in [1,2,3,4,5]:
            trained_model = pickle.load(open(training_path+"s{0}/rs{1}/rs{1}_trained_models.pkl".format(sets, random_state), "rb"))
            trained_models.append(trained_model)
            val_probas, val_preds = validate(dataset, trained_model, random_state, labels)
            proba_df = pd.DataFrame(index=dataset.index)
            pred_df = pd.DataFrame(index=dataset.index)
            for label in labels:
                proba_df[label] = val_probas[label]
                pred_df[label] = val_preds[label]
            proba_df.index.name = "id"
            pred_df.index.name = "id"
            proba_df.to_csv("validation/{0}/s{1}/{0}_rs{2}_val_probabilities.csv".format(cohort, sets, random_state))
            pred_df.to_csv("validation/{0}/s{1}/{0}_rs{2}_val_predictions.csv".format(cohort, sets, random_state))
            proba_dfs.append(proba_df)

        # find mean probabilities for cv seeds
        test_mean_proba = pd.DataFrame(index=proba_df.index)
        for label in labels: # labels (3)
            model_mean_probs = []
            for i,r in enumerate(proba_df[label]): # samples (87)
                probs = []
                for j,probadf in enumerate(proba_dfs): # seed (5)
                    probs.append(probadf[label].iloc[i])
                mean_probs = np.mean(probs)
                model_mean_probs.append(mean_probs)
            test_mean_proba[label] = model_mean_probs

        #test_mean_proba.to_csv("validation/{0}/s{1}/{0}_mean_test_probabilities.csv".format(cohort, sets))

        final_pred = []
        for r in test_mean_proba.iloc:
            if r["avg"] > 1-r["avg"]: # prob for 1 rather than 0
                final_pred.append(1)
            else:
                final_pred.append(0)
        test_mean_proba["pred"] = final_pred
        if no_truth == False:
            test_mean_proba["truth"] = truth["truth"].to_list()
        mean_tests.append(list(test_mean_proba["avg"]))
        test_mean_proba.to_csv("validation/{0}/{0}_s{1}_mean_val_probabilities.csv".format(cohort, sets))

        #auc = roc_auc_score(test_mean_proba.truth, test_mean_proba[label])
        #aucs.append()
    mean_avg_prob = pd.DataFrame(index=test_mean_proba.index)
    mean_avg_prob["avg"] = np.mean(np.array(mean_tests), axis=0)
    final_pred = []
    for r in mean_avg_prob.iloc:
        if r["avg"] > 1-r["avg"]: # prob for 1 rather than 0
            final_pred.append(1)
        else:
            final_pred.append(0)
    mean_avg_prob["pred"] = final_pred
    if no_truth == False:
        mean_avg_prob["truth"] = truth["truth"].to_list()
    mean_avg_prob.to_csv("validation/{0}/{0}_final_mean_val_probabilities.csv".format(cohort))

    if no_truth == False:
        #mets, cm = metrics_tests(mean_avg_prob["truth"], mean_avg_prob["pred"])
        mets, cm = metrics_tests(mean_avg_prob)
        mets.to_csv("validation/{0}/{0}_final_metrics.csv".format(cohort))

        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, square=True, cmap="crest")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.tight_layout()
        plt.savefig("validation/{0}/{0}_confusion_matrix.png".format(cohort), format="png", dpi=300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("training_path")
    parser.add_argument("truth_file", nargs="?", default=None)
    args = parser.parse_args()
    main(args.dataset_file, args.training_path, args.truth_file)
    print("\nDone! :)")