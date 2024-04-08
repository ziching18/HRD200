## import functions
from modules import *

def main(run, dataset_file, categories_file):
    plotStyle()
    sns.set_palette(sns.color_palette("Spectral"))
    dataset = pd.read_csv(dataset_file, index_col="id") 
    categories = pd.read_csv(categories_file, index_col="id")

    ## define variables
    pa = "" # path
    random_states = [1,2,3,4,5]
    labels = {
        "svc": "Support Vector Machine",
        "rf": "Random Forest",
        "avg": "Composite Model"
    }

    all_result = open("results/{}_all_result.txt".format(run), "w")
    all_result.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n" %
                        ("set",
                        "svc_train_mean_auc",
                        "svc_train_median_auc",
                        "svc_train_std",
                        "rf_train_mean_auc",
                        "rf_train_median_auc",
                        "rf_train_std",
                        "avg_train_mean_auc",
                        "avg_train_median_auc",
                        "avg_train_std",
                        "svc_test_mean_auc",
                        "svc_test_median_auc",
                        "svc_test_std",
                        "rf_test_mean_auc",
                        "rf_test_median_auc",
                        "rf_test_std",
                        "avg_test_mean_auc",
                        "avg_test_median_auc",
                        "avg_test_std",
                        "svc_feats",
                        "rf_feats",
                        "unique_feats"))

    ### cross-validation
    ## training
    #print("---TRAINING---\n")
    for sets in [1,2,3,4,5]:
        refits = []
        train_each_seed = []
        test_each_seed = []
        all_features = []
        features = []
        
        all_result_list = []
        all_result_list.append(str(sets))
        print("\nSET%d" % sets)
        X_train, X_test, y_train, y_test, train_ids, test_ids = split_data(dataset, categories, sets)

        for random_state in random_states:
            print("\nSEED%d" % random_state)
            # training
            trained = False
            if (path.exists(pa+"output/s{}/rs{}/rs{}_trained_refits.pkl".format(sets, random_state, random_state))):
                trained = True     # model exists, no training
            else:
                trained = False      # model does not exist, need training

            splits = defineSplits(X_train, y_train, random_state) 

            if trained == False:
                print("Model has not been trained.")
                
                # search for best parameters // returns RScv object
                trained_models = run_all_models(X_train, y_train, splits)
                pickle.dump(trained_models, open(pa+"output/s{}/rs{}/rs{}_trained_models.pkl".format(sets, random_state, random_state), "wb"))
                
                # train model with best parameters
                trained_refits, preds, trufs, pred_ids = refit_all_models(X_train, y_train, trained_models, splits, sets, random_state, labels, train_ids)
                pickle.dump(trained_refits, open(pa+"output/s{}/rs{}/rs{}_trained_refits.pkl".format(sets, random_state, random_state), "wb"))

                #refits.append(trained_models)
                refits.append(trained_refits)
            else:
                print("Model has been trained.")
                trained_models = pickle.load(open(pa+"output/s{}/rs{}/rs{}_trained_models.pkl".format(sets, random_state, random_state), "rb"))
                trained_refits, preds, trufs, pred_ids = refit_all_models(X_train, y_train, trained_models, splits, sets, random_state, labels, train_ids)
                #refits.append(trained_models)
                refits.append(trained_refits)

            dc = trained_models["svc"].best_estimator_.named_steps["dropcoll"]
            dcfit = dc.fit(X_train, y_train)
            dropped = dcfit.transform(X_train)
            all_features.append(dropped)
            f = open(pa+"output/s{}/rs{}/rs{}_dropcoll_features.txt".format(sets, random_state, random_state), "w")
            for a in dropped.columns:
                f.write("%s\n" % a)
            f.close()

            # get used feature names
            ft = []
            for label in ["svc", "rf"]:
                f = open(pa+"output/s{}/rs{}/rs{}_{}_features.txt".format(sets, random_state, random_state, label), "w")
                df1 = trained_models[label].best_estimator_.named_steps["dropcoll"].transform(X_train)
                df2 = trained_models[label].best_estimator_.named_steps["scaler"].transform(df1)
                df3 = pd.DataFrame(df2, index=df1.index, columns=df1.columns)
                kb = trained_models[label].best_estimator_.named_steps["kbest"].fit(df2, y_train)
                df4 = kb.transform(df2)
                uf = kb.get_feature_names_out(df3.columns)
                ft.append(uf)
                f.write(label + "\n")
                for u in uf:
                    f.write(u + "\n")
                f.close()
            features.append(ft)

            y_truths = []
            y_ids = []
            for i,(tr,ts) in enumerate(splits): 
                y_truths.extend(trufs["svc"][i])
                y_ids.extend(pred_ids["svc"][i])
            train_df = pd.DataFrame(index=y_ids)
            for label in labels:
                y_preds = []
                for i,(tr,ts) in enumerate(splits): 
                    y_preds.extend(preds[label][i])
                train_df[label] = y_preds
            train_df["truth"] = y_truths
            train_df.index.name = "id"
            train_df.sort_values(by="id", inplace=True)
            train_df.to_csv(pa+"output/s{}/rs{}/rs{}_train_probabilities.csv".format(sets, random_state, random_state))
            train_each_seed.append(train_df)

            # testing
            test_result, res = test_all_models(X_test, y_test, refits[random_state-1], sets, random_state, labels)
            test_df = pd.DataFrame(index=test_ids)
            for label in labels:
                test_df[label] = test_result[label]
            test_df["truth"] = list(y_test)
            test_df.index.name = "id"
            test_df.to_csv(pa+"output/s{}/rs{}/rs{}_test_probabilities.csv".format(sets, random_state, random_state))
            test_each_seed.append(test_df)

        # get mean training probabilities
        train_mean_proba = pd.DataFrame(index=train_df.index)
        for label in labels: # labels (3)
            model_mean_probs = []
            for i,r in enumerate(train_df[label]): # samples (79)
                probs = []
                for j,traindf in enumerate(train_each_seed): # seed (5)
                    probs.append(traindf[label].iloc[i])
                mean_probs = np.mean(probs)
                model_mean_probs.append(mean_probs)
            train_mean_proba[label] = model_mean_probs
        train_mean_proba["truth"] = train_df.truth
        train_mean_proba.to_csv(pa+"output/s{}/mean_train_probabilities.csv".format(sets))

        # get mean training AUCs
        f = open(pa+"output/s{}/mean_train_aucs.txt".format(sets), "w")
        for label in labels: # labels (3)
            mean_fpr = np.linspace(0, 1, 10)
            aucs = []
            for j,traindf in enumerate(train_each_seed): # seed (5)
                fpr, tpr, thresholds = roc_curve(traindf.truth, traindf[label])
                auc = roc_auc_score(traindf.truth, traindf[label])
                aucs.append(auc)  
            #mean_tpr = np.mean(tprs, axis=0)
            mfpr, mtpr, thresholds = roc_curve(train_mean_proba.truth, train_mean_proba[label])
            mean_auc = np.mean(aucs)
            all_result_list.append(mean_auc)
            median_auc = np.median(aucs)
            all_result_list.append(median_auc)
            std_auc = np.std(aucs)
            all_result_list.append(std_auc)
            f.write("{}: {}\n".format(label, mean_auc))
        f.close()

        # get mean testing probabilities
        test_mean_proba = pd.DataFrame(index=test_df.index)
        for label in labels: # labels (3)
            model_mean_probs = []
            for i,r in enumerate(test_df[label]): # samples (25)
                probs = []
                for j,testdf in enumerate(test_each_seed): # seed (5)
                    probs.append(testdf[label].iloc[i])
                mean_probs = np.mean(probs)
                model_mean_probs.append(mean_probs)
            test_mean_proba[label] = model_mean_probs

        test_mean_proba["truth"] = test_df.truth
        test_mean_proba.to_csv(pa+"output/s{}/mean_test_probabilities.csv".format(sets))

        # get mean testing AUCs
        f = open(pa+"output/s{}/mean_test_aucs.txt".format(sets), "w")
        for label in labels: # labels (3)
            mean_fpr = np.linspace(0, 1, 10)
            aucs = []
            for j,testdf in enumerate(test_each_seed): # seed (5)
                fpr, tpr, thresholds = roc_curve(testdf.truth, testdf[label])
                auc = roc_auc_score(testdf.truth, testdf[label])
                aucs.append(auc)  
            #mean_tpr = np.mean(tprs, axis=0)
            mfpr, mtpr, thresholds = roc_curve(test_mean_proba.truth, test_mean_proba[label])
            mean_auc = np.mean(aucs)
            all_result_list.append(mean_auc)
            median_auc = np.median(aucs)
            all_result_list.append(median_auc)
            std_auc = np.std(aucs)
            all_result_list.append(std_auc)
            f.write("{}: {}\n".format(label, mean_auc))
        f.close()

        allfeatures = []
        for i,label in enumerate(["svc","rf"]): # labels (3)
            feats = np.unique(np.concatenate((features[0][i], features[1][i], features[2][i], features[3][i], features[4][i])))
            allfeatures.extend(feats)
            f = open(pa+"output/s{}/{}_features.txt".format(sets,label), "w")
            for a in feats:
                f.write("%s\n" % a)
            f.close()
            all_result_list.append(len(feats))

        unique_feats = np.unique(allfeatures)
        f = open(pa+"output/s{}/unique_features.txt".format(sets), "w")
        for a in unique_feats:
            f.write("%s\n" % a)
        f.close()
        all_result_list.append(len(unique_feats))

        for ard in all_result_list:
            all_result.write(str(ard) + "\t")
        all_result.write("\n")
    all_result.close()

    df = pd.read_csv("results/{}_all_result.txt".format(run), sep="\t")
    df = df.filter(["set","svc_train_mean_auc","rf_train_mean_auc","avg_train_mean_auc",
                "svc_test_mean_auc","rf_test_mean_auc","avg_test_mean_auc",
                "svc_feats","rf_feats","unique_feats"])
    newdf = pd.DataFrame(columns=df.columns)
    newdf = newdf.append(df.mean(), ignore_index=True)
    newdf["set"] = run
    newdf.rename(columns={"set":"run"}, inplace=True)
    stds = df.std().to_frame().transpose().drop(columns=["svc_feats","rf_feats","unique_feats"])
    for col in stds.columns:
        newdf["%s_std" % col] = stds[col].values
    if (path.exists("results/result.csv")):
        newdf.to_csv("results/result.csv", mode="a", header=False)
    else:
        newdf.to_csv("results/result.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("dataset_file")
    parser.add_argument("categories_file")
    args = parser.parse_args()
    main(args.run, args.dataset_file, args.categories_file)
    print("\nDone! :)")
#'''