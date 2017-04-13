/*
 * likaizhen
 * AMM algorithm, "Trading Representability for Scalability: Adaptive Multi-Hyperplane Machine for Nonlinear Classification", url : "http://dl.acm.org/citation.cfm?id=2020420"
*/

#include <bits/stdc++.h>

using namespace std;

const bool ONLINE = true;


struct Sample {
	Sample() {
		lt_feature.clear();
		lt_feature.push_back(make_pair(0, 1.0));
	}
	Sample(const char *line, vector<string> &label_names, map<string, int> &label_ids) {
		lt_feature.clear();
		lt_feature.push_back(make_pair(0, 1.0));
		read_sample(line, label_names, label_ids);
	}

	bool read_sample(const char *line, vector<string> &label_names, map<string, int> &label_ids) {
		int len = strlen(line);
		char str[len + 1];
		strcpy(str, line);
		int pos = 0;
		int next_pos = 0;
		while (pos < len) {
			next_pos = pos;
			while (next_pos < len && str[next_pos] != ' ' && str[next_pos] != '\t')
				next_pos++;
			if (next_pos > pos) {
				for (int i = pos; i < next_pos; i++)
					label_name[i - pos] = str[i];
				label_name[next_pos - pos] = '\0';
				if (label_ids.find(label_name) == label_ids.end()) {
					label_ids[label_name] = label_names.size();
					label_names.push_back(label_name);
				}
				label_id = label_ids[label_name];
				pos = next_pos + 1;
				break;
			} else
				pos = next_pos + 1;
		}
		while (pos < len) {
			next_pos = pos;
			while (next_pos < len && str[next_pos] != ' ' && str[next_pos] != '\t')
				next_pos++;
			if (next_pos > pos) {
				str[next_pos] = '\0';
				int idx;
				double value;
				sscanf(str + pos, "%d:%lf", &idx, &value);
				lt_feature.push_back(make_pair(idx, value));
				//feature_cnt = max(feature_cnt, idx);
			}
			pos = next_pos + 1;	
		}
		if (lt_feature.size() > 1)
			lt_feature.sort();
		//printf("label name : %s\tlabel_id = %d\n", label_name, label_id);	
		return true;
	}

	char label_name[10];
	int label_id;
	int sample_id;
	int z;
	list<pair<int, double>> lt_feature;
	//map<int, double> mp_feature;
};

struct Data_sets {
	Data_sets() {
		clear();
	}
	Data_sets(const int &label_cnt, const int &feature_cnt) {
		init(label_cnt, feature_cnt);
	}
	void init(const int &label_cnt, const int &feature_cnt) {
		this -> label_cnt = label_cnt;
		this -> feature_cnt = feature_cnt;
		clear();
		train_label_cnt.resize(label_cnt, 0);
	}
	void clear() {
		label_names.clear();
		label_ids.clear();
		for (auto p : samples)
			if (NULL != p)
				delete p;
		samples.clear();
		train_label_cnt.clear();	
	}
	~Data_sets() {
		clear();
	}
	int label_cnt;
	int feature_cnt;	
	list<Sample*> samples;
	vector<string> label_names;
	map<string, int> label_ids;
	vector<int> train_label_cnt;
};

void read_training_samples(const char *train_file_path, const int &label_cnt, const int &feature_cnt, Data_sets &data_set)
{
	ifstream ifs;
	ifs.open(train_file_path, ifstream::in);
	//freopen(train_file_path, "r", stdin);
	int line_cnt = 0;
	data_set.init(label_cnt, feature_cnt);
	//train_label_cnt.resize(label_cnt, 0);
	string line;
	while (getline(ifs, line)) {
		//cerr << line << endl;	
		Sample *sample = new Sample();
		sample -> sample_id = line_cnt;
		sample -> read_sample(line.c_str(), data_set.label_names, data_set.label_ids);
		data_set.samples.push_back(sample);
		//cerr << sample -> label_name;
		//for (auto x : sample -> lt_feature)
		//	cerr << " " << x.first << "|" << x.second;
		//cerr << endl;	
		data_set.train_label_cnt[sample -> label_id]++;
		line_cnt++;
		if (line_cnt % 100000 == 0)
			cerr << "read " << line_cnt << " training samples!" << endl;
		//if (line_cnt >= 10)
		//	break;
	}
	for (int i = 0; i < label_cnt; i++) {
		cerr << data_set.label_names[i] << "\t" << data_set.train_label_cnt[i] << endl;
	}
}

class Model {
	public:
	struct Weight_node {
		Weight_node() {
			factor = 1.0;
		}	
		Weight_node* find_next(Weight_node* p) {
			if (NULL == p)
				return NULL;
			else if (!(p -> is_del))
				return p;
			else 
				return p -> next = find_next(p -> next);
		}
		int idx;
		double value;
		double factor = 1.0;
		bool is_del;
		Weight_node* next;
	};
	Model() {
		online = ONLINE;
		//init("amm.model");
	}
	void init_for_training(const int &label_cnt, const int &feature_cnt, const bool online, vector<string> &label_names, map<string, int> &label_ids, const double &lambda) {
		clear();
		this -> _t = 0;
		this -> label_cnt = label_cnt;
		this -> feature_cnt = feature_cnt;
		this -> online = online;
		this -> label_names = label_names;
		this -> label_ids = label_ids;
		this -> lambda = lambda;
		weights.resize(label_cnt);
		for (int i = 0; i < label_cnt; i++) {
			weights[i].resize(20);
			for (int j = 0; j < weights[i].size(); j++) {
				weights[i][j].resize(feature_cnt + 1);
				for (int k = 0; k <= feature_cnt; k++) {
					weights[i][j][k].value = 0.0;
					weights[i][j][k].factor = 1.0;
					weights[i][j][k].idx = k;
					weights[i][j][k].is_del = false;
					weights[i][j][k].next = (k < feature_cnt ? &weights[i][j][k + 1] : NULL);
				}
			}
		}
	}
	double cal_score(list<pair<int, double>> &lt_feature, vector<Weight_node> &vt_weight_node) {
		double sum = 0;	
		for (auto x : lt_feature) {
			if (false == vt_weight_node[x.first].is_del)
				sum += vt_weight_node[0].factor * vt_weight_node[x.first].value * x.second; 
		}
		return sum;
	}
	pair<double, int> cal_g(list<pair<int, double>> &lt_feature, vector<vector<Weight_node>> &weight_groups) {
		int weight_group_id = -1;
		double score = 0.0;
		for (int i = 0; i < weight_groups.size(); i++) {
			if (weight_groups[i][0].is_del)
				continue;
			double tmp_score = cal_score(lt_feature, weight_groups[i]);
			if (-1 == weight_group_id || tmp_score > score) {
				score = tmp_score;
				weight_group_id = i;
			}
		}
		return make_pair(score, weight_group_id);
	}

	void train(Data_sets &data_set) {
		_t = 0;
		int R = (online ? 1 : 5);	
		for (int r = 0; r < R; r++) {
			for (Sample *sample: data_set.samples) {
				train_online(sample);
				if (_t % 100000 == 0)
					cerr << "process " << _t << " iterations!" << endl;
				//if (t >= 10)
				//	break;

			}
		}

	}
	void train_online(Sample *sample) {
		++_t;
		pair<double, int> pr = cal_g(sample -> lt_feature, weights[sample -> label_id]);
		sample -> z = pr.second;
		double score = pr.first;
		double enta = 1.0 / _t / lambda;
		pair<double, int> other_pr = make_pair(0.0, -1);
		int second_label = -1;
		for (int i = 0; i < label_cnt; i++) {
			if (i == sample -> label_id)
				continue;
			pair<double, int> tmp_pr = cal_g(sample -> lt_feature, weights[i]);
			if (other_pr.second == -1 || other_pr.first < tmp_pr.first) {
				other_pr = tmp_pr;
				second_label = i;
			}
		}
				
		if (0 == _t % pruning_period)
			weight_prunt(_t);
				
		if (-1 != second_label && score < 1.0 + other_pr.first) {
			if (weights[sample -> label_id][pr.second][0].factor < 0.001) {
				for (int k = 0; k < weights[sample -> label_id][pr.second].size(); k++)
					weights[sample -> label_id][pr.second][k].value *= weights[sample -> label_id][pr.second][0].factor;
				weights[sample -> label_id][pr.second][0].factor = 1.0;
			}
			if (weights[second_label][other_pr.second][0].factor < 0.001) {
				for (int k = 0; k < weights[second_label][other_pr.second].size(); k++)
					weights[second_label][other_pr.second][k].value *= weights[second_label][other_pr.second][0].factor;
				weights[second_label][other_pr.second][0].factor = 1.0;
			}
			for (auto x : sample -> lt_feature) {
				int k = x.first;
				weights[sample -> label_id][pr.second][k].value += enta * x.second / weights[sample -> label_id][pr.second][0].factor;
				weights[second_label][other_pr.second][k].value -= enta * x.second / weights[second_label][other_pr.second][0].factor;
			}
		}

		for (int i = 0; i < label_cnt; i++) {
			for (int j = 0; j < weights[i].size(); j++) {	
				if (weights[i][j][0].is_del)
					continue;
				weights[i][j][0].factor *= 1.0 - enta * lambda;	
			}
		}
	}
	void update_by_file(const char *file_name) {
		//ifstream ifs("../data/data_for_train/test.txt");
		cerr << "update by file: " << string(file_name) << endl;
		ifstream ifs(file_name);
		string line;
		int line_cnt = 0;
		int same_cnt = 0;
		int diff_cnt = 0;
		while (getline(ifs, line)) {
			//cerr << line << endl;
			Sample sample;
			sample.read_sample(line.c_str(), label_names, label_ids);
			train_online(&sample);
			line_cnt++;
			if (line_cnt % 100000 == 0)
				cerr << "process " << line_cnt << " lines" << endl;	
		}
		ifs.close();	
		//printf("sample label id : %d\ttpredict label id = %d\n", sample.label_id, pr.first);
		//printf("sample label : %s\tpredict label : %s\n", sample.label_name, pr2.first.c_str());
		
	}

	void weight_prunt(const int &t) {
		vector<int> weight_cnt(label_cnt, 0);
		vector<pair<double, pair<int, int>>> norms;
		for (int i = 0; i < label_cnt; i++) {
			for (int j = 0; j < weights[i].size(); j++) {
				if (weights[i][j][0].is_del)
					continue;
				weight_cnt[i]++;
				double l2_square = get_l2_square(weights[i][j]);
				norms.push_back(make_pair(l2_square, make_pair(i, j)));
			}
		}
		sort(norms.begin(), norms.end());
		int cnt = 0;
		double sum;
		double bound = pruning_constant / (t - 1) / lambda;
		for (auto x : norms) {
			if (weight_cnt[x.second.first] <= 1) {
				break;
			}
			sum += x.first;
			cnt += feature_cnt + 1;
			if (sum >= bound * bound)
				break;
			else {
				for (int k = 0; k <= feature_cnt; k++) {
					weights[x.second.first][x.second.second][k].is_del = true;
				}
			}
		}
	}
	double get_l2_square(vector<Weight_node> &weight) {
		double ans = 0.0;
		for (auto x : weight)
			ans += x.value * x.value * weight[0].factor * weight[0].factor;
		return ans;
	}
	void clear() {	
		label_cnt = 0;
		feature_cnt = 0;
		label_names.clear();
		label_ids.clear();
		weights.clear();
	}
	void read_model(const char *model_file) {
		char s[100];
		FILE *pFile = fopen(model_file, "r");
		fscanf(pFile, "%d%d", &label_cnt, &feature_cnt);
		label_names.resize(label_cnt);
		for (int i = 0; i < label_cnt; i++) {
			fscanf(pFile, "%s", s);
			label_names[i] = string(s);
			label_ids[label_names[i]] = i;
		}	
		weights.resize(label_cnt);
		for (int i = 0; i < label_cnt; i++) {
			int b;
			fscanf(pFile, "%d", &b);
			weights[i].resize(b);
			//cerr << b << endl;
			for (int j = 0; j < b; j++) {
				int len, idx;
				double v;
				fscanf(pFile, "%d", &len);
				//cerr << "len = " << len << endl;
				weights[i][j].resize(feature_cnt + 1);
				for (int k = 0; k <= feature_cnt; k++) {
					weights[i][j][k].factor = 1.0;
					weights[i][j][k].value = 0.0;
					weights[i][j][k].idx = k;
					weights[i][j][k].is_del = false;
					weights[i][j][k].next = (k < feature_cnt ? &weights[i][j][k + 1] : NULL);
				}
				while (len--) {
					fscanf(pFile, " %d:%lf", &idx, &v);
					//cerr << idx << "|" << v << endl;
					weights[i][j][idx].value = v;
					weights[i][j][idx].is_del = false;
				}	
			}
		}
		fscanf(pFile, "%d", &_t);
		fclose(pFile);
		cerr << "read model:" << endl;
		cerr << "label cnt = " << label_cnt << endl;
		for (int i = 0; i < label_cnt; i++) {
			printf("%s ", label_names[i].c_str());
		}
		cerr << endl;
		for (auto x : label_ids)
			cerr << x.first << " " << x.second << endl;
		cerr << "read model end!" << endl;
	}

	void write_model(const char *model_file) {
		ofstream ofs(model_file);
		ofs << label_cnt << endl;
		ofs << feature_cnt << endl;
		for (auto x : label_names)
			ofs << x << " ";
		ofs << endl;
		for (int i = 0; i < label_cnt; i++) {
			int cnt = 0;
			for (int j = 0; j < weights[i].size(); j++) {
				if (false == weights[i][j][0].is_del)
					cnt++;
			}
			ofs << cnt << endl;
			for (int j = 0; j < weights[i].size(); j++) {
				if (weights[i][j][0].is_del)
					continue;
				ofs << feature_cnt + 1;
				for (int k = 0; k <= feature_cnt; k++)
					ofs << " " << k << ":" << weights[i][j][k].value * weights[i][j][0].factor;
				ofs << endl;
			}
		}
		ofs << _t << endl;
		ofs.close();
	}
	
	pair<int, double> predict_labelid(list<pair<int, double>> &lt_feature) {
		double score = 0;
		int idx = -1;
		for (int i = 0; i < label_cnt; i++) {
			int weight_id = -1;
			double g = 0.0;
			for (int j = 0; j < weights[i].size(); j++) {
				if (weights[i][j][0].is_del)
					continue;
				double sum = 0;
				for (auto x : lt_feature) {
					if (false == weights[i][j][x.first].is_del)
						sum += x.second * weights[i][j][x.first].value * weights[i][j][0].factor;
				}	
				if (-1 == weight_id|| g < sum) {
					weight_id = j;
					g = sum;
				}
			}
			if (-1 == idx || g > score) {
				idx = i;
				score = g;
			}
		}
		return make_pair(idx, score);
	}
	pair<string, double> predict_label(list<pair<int, double>> &lt_feature) {
		pair<int, double> pr = predict_labelid(lt_feature);
		return make_pair(label_names[pr.first], pr.second);
	}
						
	int get_label_id(const string &label) {
		return label_ids[label];
	}
	void set_pruning_constant(const double &pruning_constant) {
		this -> pruning_constant = pruning_constant;
	}

	bool online;
	int label_cnt;
	int feature_cnt;
	double lambda = 0.00001;
	int pruning_period = 10000;
	double pruning_constant = 5.0;
	int _t;
	vector<string> label_names;
	map<string, int> label_ids;
	vector<vector<vector<Weight_node>>> weights;
	vector<vector<int>> weight_cnts;
	vector<vector<bool>> weight_prunts;
}model;

int train_process()
{
	char train_file_path[] = "../data/data_for_train/train.txt";
	int label_cnt = 7;
	int feature_cnt = 40000;
	double lambda = 0.005;
	double pruning_constant = 2.0;
	int pruning_period = 10000;

	int T;
	string model_file = "amm.model";

	//Data_sets data_set;
	vector<string> label_names{"1", "2", "3", "4", "5", "6", "7"};
	map<string, int> label_ids;
	for (int i = 0; i < label_names.size(); i++)
		label_ids[label_names[i]] = i;
	time_t t_start, t_end;
	t_start = time(NULL);
	model.init_for_training(label_cnt, feature_cnt, ONLINE, label_names, label_ids, lambda);
	model.set_pruning_constant(pruning_constant);
	time_t t1 = time(NULL); 
	cerr << t1 - t_start << "(s) : start to read data and train model!" << endl;
	model.update_by_file(train_file_path);
	time_t t2 = time(NULL);
	cerr << t2 - t1 << "(s) : start to write model file!" << endl;	
	//write_model_file();
	model.write_model("amm.model");
	t_end = time(NULL);
	cerr << t_end - t2 << "(s) : write model file done!" << endl;
	cerr << t_end - t_start << "(s) : total time used!" << endl;	
	return 0;
}



void check_file(const char *file_name)
{
	//ifstream ifs("../data/data_for_train/test.txt");
	cerr << "process file: " << string(file_name) << endl;
	ifstream ifs(file_name);
	string line;
	int line_cnt = 0;
	int same_cnt = 0;
	int diff_cnt = 0;
	while (getline(ifs, line)) {
		//cerr << line << endl;
		Sample sample;
		sample.read_sample(line.c_str(), model.label_names, model.label_ids);
		pair<int, double> pr = model.predict_labelid(sample.lt_feature);
		pair<string, double> pr2 = model.predict_label(sample.lt_feature);
		//cerr << "sample label_id = " << pr.first << "\tscore = " << pr.second << endl;
		if (strcmp(sample.label_name, pr2.first.c_str()) == 0)
		//if (sample.label_name == pr2.first)
		//if (sample.label_id == pr.first)
			same_cnt++;
		else 
			diff_cnt++;
		line_cnt++;
		//printf("sample label id : %d\ttpredict label id = %d\n", sample.label_id, pr.first);
		//printf("sample label : %s\tpredict label : %s\n", sample.label_name, pr2.first.c_str());
		if (line_cnt % 100000 == 0)
			cerr << "process " << line_cnt << " lines" << endl;	
		//if (line_cnt >= 10)
		//	break;
	}
	cerr << "same_cnt = " << same_cnt << endl;
	cerr << "diff_cnt = " << diff_cnt << endl;
	cerr << "accuracy = " << (double)same_cnt / (same_cnt + diff_cnt) << endl; 
	printf("%.6f\n", (double)same_cnt / (same_cnt + diff_cnt));
}


int check_process()
{
	model.read_model("amm.model");	
	//model.write_model("model_check.txt");
	check_file("../data/data_for_train/test.txt");
	check_file("../data/data_for_train/train.txt");
	return 0;
}

int main()
{
	train_process();
	check_process();
	return 0;
}
