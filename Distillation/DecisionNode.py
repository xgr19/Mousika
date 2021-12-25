# encoding: utf-8



class DecisionNode:

    def __init__(self, feature=-1, threshold=None, label=None, label_dict=None, true_branch=None,
                 false_branch=None):
        self.feature = feature
        self.threshold = threshold
        # 'if label is not None,then it's a leaf node'
        self.label_dict = label_dict
        self.label = label
        self.true_branch = true_branch
        self.false_branch = false_branch

    def set_branch(self, true_branch, false_branch):

        self.true_branch = true_branch
        self.false_branch = false_branch

    def find_path(self, data_vec, path, feature_attr, attr_map):
        if self.label is not None:
            return path
        else:
            if feature_attr[self.feature] == "c":
                path.append([self.feature+1, self.threshold, data_vec[self.feature]])
                if data_vec[self.feature] < self.threshold:
                    path[-1].append(-1)
                    return self.false_branch.find_path(data_vec, path, feature_attr, attr_map)
                else:
                    path[-1].append(1)
                    return self.true_branch.find_path(data_vec, path, feature_attr, attr_map)
            else:
                path.append([self.feature+1, attr_map[str(self.feature) + "_" + str(int(self.threshold))],
                             attr_map[str(self.feature) + "_" + str(int(data_vec[self.feature]))]])
                if data_vec[self.feature] != self.threshold:
                    path[-1].append(-1)
                    return self.false_branch.find_path(data_vec, path, feature_attr, attr_map)
                else:
                    path[-1].append(1)
                    return self.true_branch.find_path(data_vec, path, feature_attr, attr_map)
    def show_tree(self,path, feature_attr, attr_map,model_path):
        if self.label is not None:
            path.append(' then '+str(self.label)+'\n')
            #print(path)
            with open(model_path, 'a') as f:
                message =''.join(path)
               # print(message)
                f.write(message)
            path.pop()
            return path
        else:
            if feature_attr[self.feature] == "c":
                path.append(' if feature_'+ str(self.feature)+'<'+str(self.threshold))
                #左递归
                self.false_branch.show_tree(path, feature_attr, attr_map,model_path)
                path.pop()

                path.append(' if feature_' + str(self.feature) + '>=' + str(self.threshold))
                # 右递归
                self.true_branch.show_tree(path, feature_attr, attr_map,model_path)
                path.pop()
            else:
                path.append(' if feature_' + str(self.feature) + '!=' + str(self.threshold))
                # 左递归
                self.false_branch.show_tree(path, feature_attr, attr_map,model_path)
                path.pop()

                path.append(' if feature_' + str(self.feature) + '=' + str(self.threshold))
                # 右递归
                self.true_branch.show_tree(path, feature_attr, attr_map,model_path)
                path.pop()

