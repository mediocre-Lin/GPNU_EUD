
#### 将数据分成 训练集和测试集


```python
train_data = data_all[data_all['train']==1]
test_x = data_all[data_all['train']==0]
label=train_df['Survived'].values
train_data.drop(['train','Survived'],axis=1,inplace=True)
test_x.drop(['train','Survived'],axis=1,inplace=True)
```


```python
np.save('./result/label',y)
```


```python
from sklearn.model_selection import train_test_split
feature=train_data.values
##将数据分成 训练集和测试集 ：比例为 8：2
X_train, X_test, Y_train, Y_test = train_test_split(features, y, test_size=0.2, random_state=seed)
```

## 模型建立


```python
def Titanicmodel(clf,features,test_data,y,model_name):
    if model_name =='LinearSVC':
            num_classes = 1 #类别数
    else:
            num_classes = 2 #类别数
    num_fold = 10  #10折
    fold_len = features.shape[0] // num_fold #每一折的数据量
    skf_indices = []
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed) #将训练集分为10折
    for i, (train_idx, valid_idx) in enumerate(skf.split(np.ones(features.shape[0]), y)):
        skf_indices.extend(valid_idx.tolist())
    
    train_pred = np.zeros((features.shape[0], num_classes)) #在训练集上的预测结果 (train_samples,classes)
    test_pred = np.zeros((test_data.shape[0], num_classes))#在测试集上的预测结果 (test_samples,classes)


    for fold in tqdm(range(num_fold)):


        fold_start = fold * fold_len
        fold_end = (fold + 1) * fold_len
        if fold == num_fold - 1:
            fold_end = train_data.shape[0]
        #训练部分索引 9折
        train_indices = skf_indices[:fold_start] + skf_indices[fold_end:]
        # 验证部分索引 1折
        test_indices = skf_indices[fold_start:fold_end]

        #训练部分数据 9折
        train_x = features[train_indices]
        train_y = y[train_indices]
        #验证部分数据 1折
        cv_test_x = features[test_indices]

        clf.fit(train_x, train_y) #训练

        if model_name =='LinearSVC':
            pred = clf.decision_function(cv_test_x) #在验证部分数据上 进行验证
            train_pred[test_indices] = (pred).reshape(len(pred),1) #把预测结果先通过softmax转换为概率分布(归一化) 赋给验证部分对应的位置  循环结束将会得到整个训练集上的预测结果
            pred = clf.decision_function(test_data) #得到 当前训练的模型在测试集上的预测结果
            test_pred += pred.reshape(len(pred),1) / num_fold#对每个模型在测试集上的预测结果先通过softmax转换为概率分布，再直接取平均(10折将会有10个结果)
            
        else:
            pred = clf.predict_proba(cv_test_x) #在验证部分数据上 进行验证
            train_pred[test_indices] = pred   #把预测结果 赋给验证部分对应的位置  循环结束将会得到整个训练集上的预测结果
            pred = clf.predict_proba(test_data)  #得到 当前训练的模型在测试集上的预测结果
            test_pred += pred / num_fold  #对每个模型在测试集上的预测结果直接取平均(10折将会有10个结果)
            y_pred = np.argmax(train_pred, axis=1) #对训练集上的预测结果按行取最大值 得到预测的标签
            

    if model_name =='LinearSVC':
        y_pred = (train_pred>0).astype(np.int32).reshape(len(train_pred))
        pre = (test_pred>0).astype(np.int32).reshape(len(test_pred))
    else:
        pre = np.argmax(test_pred,axis=1)
    score = accuracy_score(y, y_pred) #和训练集对应的真实标签 accuracy_score
    print('accuracy_score:',score)
    #保存逻辑回归模型在训练集和测试集上的预测结果
    np.save('./result/{0}'.format(model_name)+'train',train_pred)
    np.save('./result/{0}'.format(model_name)+'test',test_pred)
    
    submit = pd.DataFrame({'PassengerId':np.array(range(892,1310)),'Survived':pre.astype(np.int32)})
    submit.to_csv('{0}_submit.csv'.format(model_name),index=False)
    return clf,score

```

#### 逻辑回归(LR)


```python
pipe=Pipeline([('select',PCA(n_components=0.95)), 
               ('classify', LogisticRegression(random_state = seed, solver = 'liblinear'))])
param = {
        'classify__penalty':['l1','l2'],  
        'classify__C':[0.001, 0.01, 0.1, 1, 5,7,8,9,10,]}
LR_grid = GridSearchCV(estimator =pipe, param_grid = param, scoring='roc_auc', cv=5)
LR_grid.fit(features,y)
print(LR_grid.best_params_, LR_grid.best_score_)
C=LR_grid.best_params_['classify__C']
penalty = LR_grid.best_params_['classify__penalty']
LR_classify=LogisticRegression(C=C,penalty=penalty,random_state = seed, solver = 'liblinear')
LR_select =  PCA(n_components=0.95)
LR_pipeline = make_pipeline(LR_select, LR_classify)
lr_model,lr_score = Titanicmodel(LR_pipeline,feature,test_data,y,'LR')
```

    100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 105.25it/s]

    {'classify__C': 0.1, 'classify__penalty': 'l2'} 0.8643910419808494
    accuracy_score: 0.8159371492704826
    

    
    

### 支持向量机(SVM)


```python

pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify',LinearSVC(random_state=seed))])
param = {
        'select__k':list(range(20,40,2)),
        'classify__penalty':['l1','l2'],  
        'classify__C':[0.001, 0.01, 0.1, 1, 5,7,8,9,10,50,100]}
SVC_grid=GridSearchCV(estimator=pipe,param_grid=param,cv=5,scoring='roc_auc')
SVC_grid.fit(features,y)
print(SVC_grid.best_params_, SVC_grid.best_score_)
C=SVC_grid.best_params_['classify__C']
k=SVC_grid.best_params_['select__k']
penalty = SVC_grid.best_params_['classify__penalty']
SVC_classify=LinearSVC(C=C,penalty=penalty,random_state = seed)
SVC_select =  PCA(n_components=0.95)
SVC_pipeline = make_pipeline(SVC_select, SVC_classify)
SVC_model,LinearSVC_score = Titanicmodel(SVC_pipeline,feature,test_data,y,'LinearSVC')
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 51.28it/s]

    accuracy_score: 0.8215488215488216
    

    
    

### RandomForestClassifier


```python
pipe=Pipeline([('select',SelectKBest(k=34)), 
               ('classify', RandomForestClassifier(criterion='gini',
                                                   random_state = seed,
                                                   min_samples_split=4,
                                                   min_samples_leaf=5, 
                                                   max_features = 'sqrt',
                                                  n_jobs=-1,
                                                   ))])

param = {
            'classify__n_estimators':list(range(40,50,2)),  
            'classify__max_depth':list(range(10,25,2))}
rfc_grid = GridSearchCV(estimator = pipe, param_grid = param, scoring='roc_auc', cv=10)
rfc_grid.fit(features,y)
print(rfc_grid.best_params_, rfc_grid.best_score_)
n_estimators=rfc_grid.best_params_['classify__n_estimators']
max_depth = rfc_grid.best_params_['classify__max_depth']
rfc_classify=RandomForestClassifier(criterion='gini',
                                        n_estimators= n_estimators,
                                        max_depth=max_depth,
                                       random_state = seed,
                                       min_samples_split=4,
                                       min_samples_leaf=5, 
                                       max_features = 'sqrt')
rfc_select =  PCA(n_components=0.95)
rfc_pipeline = make_pipeline(rfc_select, rfc_classify)
rfc_model,rfc_score = Titanicmodel(rfc_pipeline,feature,test_data,y,'rfc')
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  5.42it/s]

    accuracy_score: 0.8092031425364759
    

    
    

### LightGBM


```python
pipe=Pipeline([('select',SelectKBest(k=34)), 
               ('classify', lgb.LGBMClassifier(random_state=seed,learning_rate=0.12,n_estimators=88,max_depth=16,
                                           min_child_samples=28,
                                            min_child_weight=0.0,
                                           classify__colsample_bytree= 0.8,
                                               colsample_bytree=0.4,
                                               objective='binary'
                                           
                                              ) )])

param = {'select__k':[i for i in range(20,40)]
#            'classify__learning_rate':[i/100 for i in range(20)]    
}
lgb_grid = GridSearchCV(estimator = pipe, param_grid = param, scoring='roc_auc', cv=10)
lgb_grid.fit(features,y)
print(lgb_grid.best_params_, lgb_grid.best_score_)
lgb_classify= lgb.LGBMClassifier(random_state=seed,
                                 learning_rate=0.12,
                                 n_estimators=88,
                                 max_depth=16,
                                 min_child_samples=28,
                                 min_child_weight=0.0,
                                 classify__colsample_bytree= 0.8,
                                 colsample_bytree=0.4,
                                 objective='binary'
                                )
lgb_select = PCA(n_components=0.96)
lgb_pipeline = make_pipeline(lgb_select, lgb_classify)
lgb_model,lgb_score = Titanicmodel(lgb_pipeline,feature,test_data,y,'lgb')
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.73it/s]

    accuracy_score: 0.7946127946127947
    

    
    


```python
 xgb.XGBClassifier(random_state=seed)
```




    XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
                  colsample_bynode=None, colsample_bytree=None, gamma=None,
                  gpu_id=None, importance_type='gain', interaction_constraints=None,
                  learning_rate=None, max_delta_step=None, max_depth=None,
                  min_child_weight=None, missing=nan, monotone_constraints=None,
                  n_estimators=100, n_jobs=None, num_parallel_tree=None,
                  objective='binary:logistic', random_state=12345, reg_alpha=None,
                  reg_lambda=None, scale_pos_weight=None, subsample=None,
                  tree_method=None, validate_parameters=None, verbosity=None)



### Xgboost


```python
pipe=Pipeline([('select',SelectKBest(k=34)), 
               ('classify', xgb.XGBClassifier(random_state=seed,
                                              learning_rate=0.12,
                                              n_estimators=80,
                                              max_depth=8,
                                              min_child_weight=3,
                                              subsample=0.8,
                                              colsample_bytree=0.8,
                                              gamma=0.2,
                                              reg_alpha=0.2,
                                              reg_lambda=0.1,
                                             )
               )])
param = {  'select__k':[i for i in range(20,40)
           'classify__learning_rate':[i/100 for i in range(10,20)],
}
xgb_grid = GridSearchCV(estimator = pipe, param_grid = param, scoring='roc_auc', cv=10)
xgb_grid.fit(features,y)
print(xgb_grid.best_params_, xgb_grid.best_score_)
xgb_classify= xgb.XGBClassifier(random_state=seed,
                                              learning_rate=0.12,
                                              n_estimators=80,
                                              max_depth=8,
                                              min_child_weight=3,
                                              subsample=0.8,
                                              colsample_bytree=0.8,
                                              gamma=0.2,
                                              reg_alpha=0.2,
                                              reg_lambda=0.1,
                                             )
xgb_select =  SelectKBest(k = 34)
xgb_pipeline = make_pipeline(xgb_select, xgb_classify)
xgb_model,xgb_score = Titanicmodel(xgb_pipeline,'xgb')
```

     10%|████████▎                                                                          | 1/10 [00:00<00:01,  8.85it/s]

    {'select__k': 34} 0.8930989729225024
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.06it/s]

    accuracy_score: 0.8338945005611672
    

    
    

### 模型融合


```python
LR_train = np.load('./result/LRtrain.npy')
LR_test = np.load('./result/LRtest.npy')
LinearSVC_train = np.load('./result/LinearSVCtrain.npy')
LinearSVC_test = np.load('./result/LinearSVCtest.npy')
rfc_train = np.load('./result/rfctrain.npy')
rfc_test = np.load('./result/rfctest.npy')
xgb_train = np.load('./result/xgbtrain.npy')
xgb_test = np.load('./result/xgbtest.npy')
lgb_train = np.load('./result/lgbtrain.npy')
lgb_test= np.load('./result/lgbtest.npy')
label = np.load('./result/label.npy')
```


```python
train_data = ( LR_train, rfc_train, LinearSVC_train,xgb_train, lgb_train)
test_x = ( LR_test, rfc_test, LinearSVC_test,xgb_test, lgb_test)
train_data = np.hstack(train_data)
test_x = np.hstack(test_x)
```


```python
model = LogisticRegression(random_state=seed)
lgbm_7leaves_model,lgbm_7leaves_score = Titanicmodel(model,features=train_data,test_data=test_x,y=label,model_name='lr_stacking')

```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 82.64it/s]

    accuracy_score: 0.8361391694725028
    

 
