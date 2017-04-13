/*
 * likaizhen@youku.com
 * AMM-rank algorithm, "Non-linear Label Ranking for Large-scale Prediction of Long-Term User Interests", url: "http://arxiv.org/abs/1606.08963"
 * 
*/

#include <bits/stdc++.h>

using namespace std;

const bool ONLINE = true;


struct Sample {
	Sample() {
		lt_feature.clear();
		lt_feature.push_back(make_pair(0, 1.0));
	}	

	void get_label_name_sort(char *str) {
		label_name_sort.clear();
		int pos = 0;
		int len = strlen(str);
		while (pos < len) {
			int next_pos = pos;
			while (next_pos < len && str[next_pos] != ',')
				next_pos++;
			//printf("pos = %d\tnext_pos = %d\n", pos, next_pos);
			str[next_pos] = '\0';
			label_name_sort.push_back((string)(str + pos));
			//cerr << label_name_sort.back() << endl;
			pos = next_pos + 1;
		}
	}
	bool read_sample(const char *line, vector<string> &label_names, map<string, int> &label_ids, const int &label_cnt) {
		int len = strlen(line);
		char str[len + 1], str2[len + 1];
		strcpy(str, line);
		int pos = 0;
		int next_pos = 0;
		while (pos < len) {
			next_pos = pos;
			while (next_pos < len && str[next_pos] != ' ' && str[next_pos] != '\t')
				next_pos++;
			if (next_pos > pos) {
				for (int i = pos; i < next_pos; i++)
					str2[i - pos] = str[i];
				str2[next_pos - pos] = '\0';
				//printf("%s\n", str2);
				get_label_name_sort(str2);
				label_id_sort.resize(label_name_sort.size());
				label_id_rank.resize(label_cnt, -1);
				for (int i = 0; i < label_name_sort.size(); i++) {
					if (label_ids.find(label_name_sort[i]) == label_ids.end()) {
						label_ids[label_name_sort[i]] = label_names.size();
						label_names.push_back(label_name_sort[i]);
					}
					label_id_sort[i] = label_ids[label_name_sort[i]];
					label_id_rank[label_id_sort[i]] = i;
				}	
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

	//char label_name[10];
	//int label_id;
	vector<string> label_name_sort;
	vector<int> label_id_rank;
	vector<int> label_id_sort;
	int sample_id;
	//int z;
	list<pair<int, double>> lt_feature;
	//map<int, double> mp_feature;
};
/*
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
*/


class AMM_rank_model {
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
	AMM_rank_model() {
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
	void make_random_weights() {
		uniform_real_distribution<double> unif(-1.0, 1.0);
		default_random_engine re;
		for (int i = 0; i < weights.size(); i++) {
			for (int j = 0; j < weights[i].size(); j++) {
				for (int k = 0; k < weights[i][j].size(); k++) {
					weights[i][j][k].value = unif(re);
					weights[i][j][k].factor = 1.0;
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
	/*
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
	*/
	void train_online(Sample *sample, vector<pair<double, int>> &prs) {
		++_t;
		for (int label_id : sample -> label_id_sort)
			prs[label_id] = cal_g(sample -> lt_feature, weights[label_id]);	
		double enta = 1.0 / _t / lambda;	
		if (0 == _t % pruning_period)
			weight_prunt(_t);
		for (int i = 0; i < sample -> label_id_sort.size(); i++) {
			int id1 = sample -> label_id_sort[i];
			for (int j = i + 1; j < sample -> label_id_sort.size(); j++) {
				int id2 = sample -> label_id_sort[j];
				if (prs[id1].first < prs[id2].first + 1.0) {
					if (weights[id1][prs[id1].second][0].factor < 0.001) {
						for (int k = 0; k <= weights[id1][prs[id1].second].size(); k++)
							weights[id1][prs[id1].second][k].value *= weights[id1][prs[id1].second][0].factor;
						weights[id1][prs[id1].second][0].factor = 1.0;
					}
					if (weights[id2][prs[id2].second][0].factor < 0.001) {
						for (int k = 0; k <= weights[id2][prs[id2].second].size(); k++)
							weights[id2][prs[id2].second][k].value *= weights[id2][prs[id2].second][0].factor;
						weights[id2][prs[id2].second][0].factor = 1.0;
					}
					for (auto x : sample -> lt_feature) {
						int k = x.first;
						weights[id1][prs[id1].second][k].value += enta * x.second / weights[id1][prs[id1].second][0].factor;
						weights[id2][prs[id2].second][k].value -= enta * x.second / weights[id2][prs[id2].second][0].factor;
					}
				}
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
		vector<pair<double, int>> prs(label_cnt);
		while (getline(ifs, line)) {
			//cerr << line << endl;
			Sample sample;
			sample.read_sample(line.c_str(), label_names, label_ids, label_cnt);
			/*
			for (auto label_id : sample.label_id_rank)
				cerr << label_id << " ";
			cerr << endl;
			for (auto label_id : sample.label_id_sort)
				cerr << label_id << " ";
			cerr << endl;
			for (auto x : sample.lt_feature)
				cerr << x.first << ":" << x.second << " ";
			cerr << endl;
			cerr << "read data done!" << endl;
			*/
			train_online(&sample, prs);
			line_cnt++;
			if (line_cnt % 100000 == 0)
				cerr << "process " << line_cnt << " lines" << endl;
			//if (line_cnt >= 10)
			//	break;	
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
	
	void predict_label_id_sort(list<pair<int, double>> &lt_feature, vector<pair<double, int>> &predict_order) {
		predict_order.resize(label_cnt);
		for (int i = 0; i < label_cnt; i++) {
			pair<double, int> pr = cal_g(lt_feature, weights[i]);
			predict_order[i].second = i;
			predict_order[i].first = pr.first;
		}
		sort(predict_order.begin(), predict_order.end());	
	}
	/*
	pair<string, double> predict_label(list<pair<int, double>> &lt_feature) {
		pair<int, double> pr = predict_labelid(lt_feature);
		return make_pair(label_names[pr.first], pr.second);
	}
	*/					
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
	int label_cnt = 30;
	int feature_cnt = 134;
	double lambda = 0.0000001;
	double pruning_constant = 5.0;
	int pruning_period = 10000;

	int T;
	string model_file = "amm_rank.model";

	//Data_sets data_set;
	vector<string> label_names{"1","2","21","5","99","a","b","c","d","e","f","g","h","j","l","m","n","o","pker","q","r","s","t","u","v","w","wdyg","wgju","z","zpai"};
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
	model.write_model(model_file.c_str());
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
	double e_dist = 0.0;
	int line_cnt = 0;
	//int same_cnt = 0;
	//int diff_cnt = 0;
	vector<pair<double, int>> prs;
	while (getline(ifs, line)) {
		//cerr << line << endl;
		Sample sample;
		sample.read_sample(line.c_str(), model.label_names, model.label_ids, model.label_cnt);
		model.predict_label_id_sort(sample.lt_feature, prs);
		double sum = 0.0;
		for (int i = 0; i < model.label_cnt; i++) {
			if (-1 == sample.label_id_rank[prs[i].second])
				continue;
			for (int j = i + 1; j < model.label_cnt; j++) {
				if (-1 == sample.label_id_rank[prs[j].second])
					continue;
				if (sample.label_id_rank[prs[i].second] < sample.label_id_rank[prs[j].second])
					sum += 1.0;
			}
		}
		sum /= sample.label_id_sort.size();
		sum /= model.label_cnt - 0.5 * (sample.label_id_sort.size() + 1);
		e_dist += sum;
		/*
		pair<int, double> pr = model.predict_labelid(sample.lt_feature);
		pair<string, double> pr2 = model.predict_label(sample.lt_feature);
		//cerr << "sample label_id = " << pr.first << "\tscore = " << pr.second << endl;
		if (strcmp(sample.label_name, pr2.first.c_str()) == 0)
		//if (sample.label_name == pr2.first)
		//if (sample.label_id == pr.first)
			same_cnt++;
		else 
			diff_cnt++;
		*/
		line_cnt++;
		//printf("sample label id : %d\ttpredict label id = %d\n", sample.label_id, pr.first);
		//printf("sample label : %s\tpredict label : %s\n", sample.label_name, pr2.first.c_str());
		if (line_cnt % 100000 == 0)
			cerr << "process " << line_cnt << " lines" << endl;	
		//if (line_cnt >= 10)
		//	break;
	}
	e_dist /= line_cnt;
	cerr << "e_dist = " << e_dist << endl;
	/*
	cerr << "same_cnt = " << same_cnt << endl;
	cerr << "diff_cnt = " << diff_cnt << endl;
	cerr << "accuracy = " << (double)same_cnt / (same_cnt + diff_cnt) << endl; 
	printf("%.6f\n", (double)same_cnt / (same_cnt + diff_cnt));
	*/
}

void check_file2(const char *file_name)
{
	//ifstream ifs("../data/data_for_train/test.txt");
	cerr << "process file: " << string(file_name) << endl;
	ifstream ifs(file_name);
	string line;
	double e_dist = 0.0;
	int line_cnt = 0;
	//int same_cnt = 0;
	//int diff_cnt = 0;
	vector<pair<double, int>> prs;
	while (getline(ifs, line)) {
		//cerr << line << endl;
		Sample sample;
		sample.read_sample(line.c_str(), model.label_names, model.label_ids, model.label_cnt);
		model.predict_label_id_sort(sample.lt_feature, prs);
		double sum = 0.0;
		for (int i = 0; i < model.label_cnt; i++) {
			if (-1 == sample.label_id_rank[prs[i].second])
				continue;
			for (int j = i + 1; j < model.label_cnt; j++) {
				/*
				if (-1 == sample.label_id_rank[prs[j].second])
					continue;
				*/
				if (-1 == sample.label_id_rank[prs[j].second] || sample.label_id_rank[prs[i].second] < sample.label_id_rank[prs[j].second])
					sum += 1.0;
			}
		}
		sum /= sample.label_id_sort.size();
		//sum /= model.label_cnt - 0.5 * (sample.label_id_sort.size() + 1);
		sum /= model.label_cnt;
		e_dist += sum;
		/*
		pair<int, double> pr = model.predict_labelid(sample.lt_feature);
		pair<string, double> pr2 = model.predict_label(sample.lt_feature);
		//cerr << "sample label_id = " << pr.first << "\tscore = " << pr.second << endl;
		if (strcmp(sample.label_name, pr2.first.c_str()) == 0)
		//if (sample.label_name == pr2.first)
		//if (sample.label_id == pr.first)
			same_cnt++;
		else 
			diff_cnt++;
		*/
		line_cnt++;
		//printf("sample label id : %d\ttpredict label id = %d\n", sample.label_id, pr.first);
		//printf("sample label : %s\tpredict label : %s\n", sample.label_name, pr2.first.c_str());
		if (line_cnt % 100000 == 0)
			cerr << "process " << line_cnt << " lines" << endl;	
		//if (line_cnt >= 10)
		//	break;
	}
	e_dist /= line_cnt;
	cerr << "e_dist = " << e_dist << endl;
	/*
	cerr << "same_cnt = " << same_cnt << endl;
	cerr << "diff_cnt = " << diff_cnt << endl;
	cerr << "accuracy = " << (double)same_cnt / (same_cnt + diff_cnt) << endl; 
	printf("%.6f\n", (double)same_cnt / (same_cnt + diff_cnt));
	*/
}
int check_process()
{
	model.read_model("amm_rank.model");	
	//model.write_model("model_check.txt");
	check_file("../data/data_for_train/test.txt");
	check_file("../data/data_for_train/train.txt");
	
	model.make_random_weights();
	check_file("../data/data_for_train/test.txt");
	check_file("../data/data_for_train/train.txt");
	
	return 0;
}
int check_process2()
{
	model.read_model("amm_rank.model");	
	//model.write_model("model_check.txt");
	check_file2("../data/data_for_train/test.txt");
	//check_file2("../data/data_for_train/train.txt");
	
	model.make_random_weights();
	check_file2("../data/data_for_train/test.txt");
	//check_file("../data/data_for_train/train.txt");
	
	return 0;
}
int main()
{
	train_process();
	//check_process();
	check_process2();
	return 0;
}
