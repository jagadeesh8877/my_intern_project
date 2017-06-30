#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
from os import listdir
from os.path import join
import re
from pattern.en import tokenize
from pattern.de import parse, split
import os
import sys
reload(sys)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

sys.setdefaultencoding("utf-8")
app = Flask(__name__)

APP__ROOT = os.path.dirname(os.path.abspath(__file__))
train_fnames = []
test_fnames = []
r_favourite = []
descend_algo = []
end_symbol_pattern = ('(.*(;|\.|\)|\).|])$)')
end_sym_regex = re.compile(end_symbol_pattern)
digit_paterns = '(^(\(|\[)?\d+(,|;|\.|]|\))?(,)?$|\d+(\.|,)\d+(\.\d+)?(,|;|\.|])?|(\()?/\d{2,4}(,|;|\.|]|\))?|^I+(V)?(,|;|\.|])?$|\d+(-|/)\d+|^\d+[a-zA-Z]+(,|;|\.|]|\))?(,)?$|^[a-zA-Z](\.)?$|([0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+))'
digit_regex = re.compile(digit_paterns)

@app.route("/")
def start2():
    return render_template("jagadeeshstart.html")
@app.route("/train")
def train():
    return render_template("jagadeeshtraining.html")
@app.route("/test")
def test():
    return render_template("jagadeeshtesting.html")
@app.route("/trainupload", methods=['POST'])
def train_upload():
    global train_fnames
    target = os.path.join(APP__ROOT, 'traindata/')
    if not os.path.isdir(target):
        os.mkdir(target)
    for file1 in request.files.getlist("train_file"):
        fname1 = file1.filename
        train_fnames.append(fname1)
        destination = "/".join([target, fname1])
        file1.save(destination)
    return redirect(url_for("train"))
@app.route("/testupload", methods=['POST'])
def test_upload():
    global test_fnames
    target = os.path.join(APP__ROOT, 'testdata/')

    if not os.path.isdir(target):
        os.mkdir(target)
    for file2 in request.files.getlist("test_file"):
        fname2 = file2.filename
        test_fnames.append(fname2)
        destination = "/".join([target, fname2])
        file2.save(destination)
    return redirect(url_for("test"))
@app.route('/f1trainselection', methods = ['POST', 'GET'])
def feature_form1():
    target = os.path.join(APP__ROOT, 'traindata/')
    f1train_path = os.path.join(APP__ROOT, 'f1train_output/')
    if not os.path.isdir(f1train_path):
        os.mkdir(f1train_path)   
    is_favourite = []   
    if request.method == 'POST':
        favourite = request.form.getlist('level_1_features')
        for item in favourite:
            is_favourite.append(str(item))       
    for y in train_fnames:
        output_file = open(join(f1train_path,y),"w")
        text_data = open(join(target,y),"r")
        lines = text_data.readlines()
        final_text = ''
        i = 0
        for x in lines:
            para = x.strip("\n")
            for t in is_favourite:      
                if t == "len_of_para":
                    feature1 = len(para)
                    final_text += str(feature1) + "\t" 
                elif t == "contains_digit":
                    feature7 = search_digit(para)
                    final_text += str(feature7) + "\t"      
                elif t == "number_of_sentences":
                    feature9 = sentence_count(para)
                    final_text += str(feature9) + "\t"
                elif t == "paranthesis_ending":
                    feature10 = paranthesis_end(para)
                    final_text += str(feature10) + "\t"
                elif t == "multiple_fullstop":
                    feature11 = multiple_stop(para)
                    final_text += str(feature11) + "\t"
                elif t == "parathesised_value":
                    feature5 = match_balanced_paranthesis(para)
                    final_text += str(feature5) + "\t"
                elif t == "consid_value":
                    feature12 = contain_consid_keyword(para)
                    final_text += str(feature12)
            final_text += "\n"
        print final_text
        output_file.write(final_text)
    success_msg = "feature files for first level are generated carry on with second level" 
    return render_template('ftrain.html', 
                        success_msg=success_msg)

@app.route('/f2trainselection', methods = ['POST', 'GET'])
def feature_train():
    global r_favourite
    target = os.path.join(APP__ROOT, 'traindata/')
    f2train_path = os.path.join(APP__ROOT, 'f2train_output/')
    if not os.path.isdir(f2train_path):
        os.mkdir(f2train_path)
    if request.method == 'POST':
        selected = request.form.getlist('level_2_features')
        for item in selected:
            r_favourite.append(str(item))  
    for t in train_fnames:
        output_file = open(join(f2train_path,t),"w")      
        text_data = open(join(target,t),"r")
        lines = text_data.readlines()
        result = ''
        for l in lines:
            para = l.strip("\n")
            sentences = tokenize(para)
            for sent in sentences:
                parsed_text = parse(sent)
                tagged_tokens = parsed_text.split()
                if len(tagged_tokens) != 1: 
                    tagged_tokens_temp = []
                    for item in tagged_tokens:
                        tagged_tokens_temp += item
                    tagged_tokens = [tagged_tokens_temp]
                tagged_tokens = tagged_tokens[0]
                for tagged_token in tagged_tokens:
                    token = tagged_token[0]
                    result += str(token) + "\n"
                    for i in r_favourite:
                        if i == "token_length":
                            token_length = len(token)
                            result += "\t" + str(token_length)
                        if i == "uppercase":
                            if token.isupper():
                                uppercase = "1" 
                            else:
                                uppercase = "0"
                            result += "\t" + str(uppercase)
                        if i == "initcase":
                            try:
                                if token[0].isupper():
                                    initcase = "1"  
                                else:
                                    initcase = "0"
                            except:
                                initcase = "0"
                            result += "\t" +str(initcase)
                        if i == "start_paranthesis":
                            if token.startswith('('):
                                start_paranthesis = "1"
                            else:
                                start_paranthesis = "0"
                            result += "\t" + str(start_paranthesis)
                        if i == "token_ending_regex":
                            if end_sym_regex.match(token) != None:
                                token_ending_regex = "1"
                            else:
                                token_ending_regex = "0"
                            result += "\t" + str(token_ending_regex)
                        if i == "digit_regex_value":
                            if digit_regex.search(token) != None:
                                digit_regex_value = "1"
                            else:
                                digit_regex_value = "0"
                            result += "\t" + str(digit_regex_value)
                        if i == "pos_tag":
                            pos_tag = tagged_token[1]
                            result += "\t" + str(pos_tag)
                        if i == "phrase_tag":
                            phrase_tag = tagged_token[2]
                            result += "\t" + str(phrase_tag)    
                    result += "\n"
        output_file.write(result)  
    success_msg ="feature files for Second level are generated,carry on with algorithms"
    return render_template('ftrain.html', 
                        success_msg=success_msg)

@app.route('/f2testselection', methods = ['POST', 'GET'])
def feature_test():
    target = os.path.join(APP__ROOT, 'testdata/')
    f2test_path = os.path.join(APP__ROOT, 'f2test_output/')
    if not os.path.isdir(f2test_path):
        os.mkdir(f2test_path)
    if request.method == 'GET':
        for t in test_fnames:        
            output_file = open(join(f2test_path,t),"w")    
            text_data = open(join(target,t),"r")
            lines = text_data.readlines()
            result = ''
            for l in lines:
                para = l.strip("\n")
                sentences = tokenize(para)
                for sent in sentences:
                    parsed_text = parse(sent)
                    tagged_tokens = parsed_text.split()
                    if len(tagged_tokens) != 1: 
                        tagged_tokens_temp = []
                        for item in tagged_tokens:
                            tagged_tokens_temp += item
                        tagged_tokens = [tagged_tokens_temp]
                    tagged_tokens = tagged_tokens[0]
                    for tagged_token in tagged_tokens:
                        token = tagged_token[0]
                        result += str(token) + "\n"
                        for i in r_favourite:
                            if i == "token_length":
                                token_length = len(token)
                                result += "\t" + str(token_length)
                            if i == "uppercase":
                                if token.isupper():
                                    uppercase = "1" 
                                else:
                                    uppercase = "0"
                                result += "\t" + str(uppercase)
                            if i == "initcase":
                                try:
                                    if token[0].isupper():
                                        initcase = "1"  
                                    else:
                                        initcase = "0"
                                except:
                                    initcase = "0"
                                result += "\t" +str(initcase)
                            if i == "start_paranthesis":
                                if token.startswith('('):
                                    start_paranthesis = "1"
                                else:
                                    start_paranthesis = "0"
                                result += "\t" + str(start_paranthesis)
                            if i == "token_ending_regex":
                                if end_sym_regex.match(token) != None:
                                    token_ending_regex = "1"
                                else:
                                    token_ending_regex = "0"
                                result += "\t" + str(token_ending_regex)
                            if i == "digit_regex_value":
                                if digit_regex.search(token) != None:
                                    digit_regex_value = "1"
                                else:
                                    digit_regex_value = "0"
                                result += "\t" + str(digit_regex_value)
                            if i == "pos_tag":
                                pos_tag = tagged_token[1]
                                result += "\t" + str(pos_tag)
                            if i == "phrase_tag":
                                phrase_tag = tagged_token[2]
                                result += "\t" + str(phrase_tag)        
                        result += "\n"
            output_file.write(result)

    return render_template('ftest.html', 
                        r_favourite=str(r_favourite))
                       
@app.route('/accuracy', methods = ['POST', 'GET'])
def ml_algos():
    global descend_algo   
    if request.method == 'POST':
        selected = request.form.getlist('accuracy')
        favourite_algo = str(selected)
        favourite_algo = favourite_algo.replace("[u'","").replace("']","").lstrip().rstrip()
        input_path = "/home/jagadeesh8877/Downloads/a_input_files/all_lang/"
        file_list = listdir(input_path)
        x_train = [["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11"]]
        y_train = []
        for file in file_list[0:10]:
            input_ = open(join(input_path,file),"r")
            content = input_.readlines()
            for line in content:
                if line.strip() != '':
                    line = line.strip("\n")
                    features = line.split("\t")
                    p = features[2:13]
                    q = features[-1]
                    a = []
                    for t in p:
                        a.append(float(t))
                    x_train.append(a)
                    if q == "B_CITATION":
                        y_train = y_train+[float("1")]
                    elif q == "I_CITATION":
                        y_train = y_train+[float("2")]
                    elif q == "O":
                        y_train = y_train+[float("0")]
        headers = x_train.pop(0)
        df_x = pd.DataFrame(x_train, columns=headers)
        df_y = pd.DataFrame({'target':y_train})
        result = pd.concat([df_x, df_y],axis=1)
        array = result.values
        X = array[:,0:-1]
        Y = array[:,-1]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        scoring = 'accuracy'
        models = []
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('SGD',SGDClassifier(loss="hinge", penalty="l2")))
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('RF',RandomForestClassifier(n_estimators=10)))
        if favourite_algo == "COMPARE_ALL":
            results = []
            names = []
            msgs = ''
            for name, model in models:                       
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                names.append(name)
                results.append(cv_results.mean())            
                msgs += "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) + "\n"
            final = dict(zip(names,results))
            final_sorted_keys = sorted(final, key=final.get, reverse=True)
            #a=[]
            for r in final_sorted_keys:
                descend_algo.append(tuple([r, final[r]]))
            print descend_algo           
            for macs in descend_algo[0]:
                if macs == "SVM":
                    best_algo = "for the given data Support Vector Machines algorithm is best "
                    svm = SVC()
                    svm.fit(X_train, Y_train)
                    svmfile = 'svmtrained_model.sav'
                    joblib.dump(svm,svmfile)
                    predictions = svm.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "RF":
                    best_algo = "for the given data RandomForestClassifier algorithm is best"
                    rf = RandomForestClassifier(n_estimators=10)
                    rf.fit(X_train, Y_train)
                    rffile = 'rftrained_model.sav'
                    joblib.dump(rf,rffile)
                    predictions = rf.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "CART":
                    best_algo = "for the given data DecisionTreeClassifier algorithm is best"
                    dt = DecisionTreeClassifier()
                    dt.fit(X_train, Y_train)
                    dtfile = 'dttrained_model.sav'
                    joblib.dump(dt,dtfile)
                    predictions = dt.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "NB":
                    best_algo = "for the given data Naive Bayes algorithm is best"
                    gnb = GaussianNB()
                    gnb.fit(X_train, Y_train)
                    gnbfile = 'gnbtrained_model.sav'
                    joblib.dump(gnb,gnbfile)
                    predictions = gnb.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "SGD":
                    best_algo = "for the given data Stochastic Gradient Descent algorithm is best"
                    sgd = SGDClassifier(loss="hinge", penalty="l2")
                    sgd.fit(X_train, Y_train)
                    sgdfile = 'sgdtrained_model.sav'
                    joblib.dump(sgd,sgdfile)
                    predictions = sgd.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "LR":
                    best_algo = "for the given data LogisticRegression algorithm is best"
                    lr =  LogisticRegression()
                    lr.fit(X_train, Y_train)
                    lrfile = 'lrtrained_model.sav'
                    joblib.dump(lr,lrfile)
                    predictions = lr.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)  
                elif macs == "LDA":
                    best_algo = "for the given data LinearDiscriminantAnalysis algorithm is best"
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(X_train, Y_train)
                    ldafile = 'ldatrained_model.sav'
                    joblib.dump(lda,ldafile)
                    predictions = lda.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo)
                elif macs == "KNN":
                    best_algo = "for the given data  KNearest Neighbors algorithm is best"
                    knn = KNeighborsClassifier()
                    knn.fit(X_train, Y_train)
                    knnfile = 'knntrained_model.sav'
                    joblib.dump(knn,knnfile)
                    predictions = knn.predict(X_validation)
                    accu_score = accuracy_score(Y_validation, predictions)
                    conf_matrix = confusion_matrix(Y_validation, predictions)
                    classi_report = classification_report(Y_validation, predictions)
                    print accu_score,conf_matrix,classi_report
                    return render_template('accuracy.html', 
                       accu_score=accu_score,
                       conf_matrix=conf_matrix,
                       classi_report=classi_report,
                       best_algo=best_algo) 

        for name, model in models:
            msg = ''
            if name == favourite_algo:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)         
                msg += "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) + "\n"          
                return render_template('accumsg.html',
                        success_msg = msg)
@app.route('/finaltest', methods = ['POST', 'GET'])
def test_algos():      
    if request.method == 'GET':
        print " i am in testing stage"
        print descend_algo[0]
        #selected = request.form.getlist('accuracy')
        #favourite_algo = str(selected)
        #favourite_algo = favourite_algo.replace("[u'","").replace("']","").lstrip().rstrip()  
        input_path = "/home/jagadeesh8877/Downloads/a_input_files/all_lang/"
        predict_path = os.path.join(APP__ROOT, 'predicted/')
        if not os.path.isdir(predict_path):
            os.mkdir(predict_path)
        output = open(join(predict_path,"fianl_output.txt"),"w")
        file_list = listdir(input_path)
        x_test = [["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11"]]
        for file in file_list[0:5]:
            input_ = open(join(input_path,file),"r")
            content = input_.readlines()
            for line in content:
                if line.strip() != '':
                    line = line.strip("\n")
                    features = line.split("\t")
                    p = features[2:13]
                    q = features[-1]
                    a = []
                    for t in p:
                        a.append(float(t))
                    x_test.append(a)   
        headers = x_test.pop(0)
        df_x = pd.DataFrame(x_test, columns=headers)
        array = df_x.values
        X_test = array[:,0:]
        cited_result = []
        for m in descend_algo[0]:
            if m == "RF":
                loaded_model = joblib.load('rftrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through RandomForestClassifier algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                
            elif m == "SVM":
                loaded_model = joblib.load('svmtrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through Support Vector Machines algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
            
            elif m == "CART":
                loaded_model = joblib.load('dttrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through DecisionTreeClassifier algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)

            elif m == "NB":
                loaded_model = joblib.load('nbtrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through Naive Bayes algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                
            elif m == "SGD":
                loaded_model = joblib.load('sgdtrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through Stochastic Gradient Descent algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                
            elif m == "LR":
                loaded_model = joblib.load('lrtrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through LogisticRegression algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                
            elif m == "LDA":
                loaded_model = joblib.load('ldatrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through LinearDiscriminantAnalysis algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                
            elif m == "KNN":
                loaded_model = joblib.load('knntrained_model.sav')
                result = loaded_model.predict(X_test)
                for r in result:    
                    if r == float("1"):
                        cited_result = cited_result+["B_CITATION"]
                    elif r == float("2"):
                        cited_result = cited_result+["I_CITATION"]
                    elif r == float("0"):
                        cited_result = cited_result+["O"]
                output.write(str(cited_result))
                print cited_result
                success_msg ="output is predicted through KNearest Neighbors algorithm"
                return render_template('final_test.html', 
                        success_msg=success_msg)
                     
def sentence_count(text):
    sentences = tokenize(text)
    sentence_count = len(sentences)
    return sentence_count
def paranthesis_end(text):
    sentences = tokenize(text)
    sent_end_flag = 0
    for sent in sentences:
        if sent.strip().endswith(').'):
            sent_end_flag = 1
    return sent_end_flag
def multiple_stop(text):            
    sentences = tokenize(text)
    multiple_delimiter_occurence = 0
    for sent in sentences:
        if sent.count('.') > 1:
            multiple_delimiter_occurence = 1
    return multiple_delimiter_occurence   
def search_digit(text):
    if re.search('\d+', text) != None:
        return "1"
    elif re.search('I+(V)?', text) != None:
        return "1"
    else:
        return "0"           
def match_balanced_paranthesis(text):
    if re.search('\(.*\)', text) != None:
        return "1"
    else:
        return "0"
def contain_consid_keyword(text):
    if "consid." in text.lower():
        return "1"
    else:
        return "0"                    

if __name__ == "__main__":
    app.run(debug=True)