#include <vector>
#include <cmath>

#include <iostream>

using namespace std;

const double inf = 1e300;

double normal_distribution(double x) {
    const double nd_coef = sqrt(M_PI * 2.);
    return exp(-0.5 * pow(x, 2)) / nd_coef;
}

double sigmoid(double x) {
    return 1. / (1. + exp(-x));
}

struct Response {
    int examinee;
    int item;
    int response;
};

struct Model {
    const int item_count;
    vector<double> discrimination, difficulty;
    Model (int ic): item_count(ic), discrimination(item_count, 1), difficulty(item_count, 0) {}

    double probability_base(double ability, double diff, double disc) {
        // 2PLM IRT
        const double D = 1.7;
        return sigmoid(D * disc * (ability - diff));
    }

    double response_probability (double ability, const Response& response) {
        int item_index = response.item;
        const double base = probability_base(ability, difficulty[item_index], discrimination[item_index]);
        return base * (response.response) + (1. - base) * (1 - response.response);
    }
    
    double response_probability (double ability, const vector<Response>& response) {
        double ans = 1.;
        for (int i = 0; i < response.size(); ++i) {
            ans *= response_probability(ability, response[i]);
        }
        return ans;
    }

};

Model estimate (const vector<Response>& responses, int item_count, int examinee_count) {
    Model model(item_count);
    vector<vector<Response> > e_resp(examinee_count, vector<Response>());
    for (int i = 0; i < responses.size(); ++i) {
        e_resp[responses[i].examinee].push_back(responses[i]);
    }

    cerr << "EM algorithm start." << endl;

    double likelihood = -inf;
    while (true) {
        cerr << "Loop start. likelihood = " << likelihood << endl;
        cerr << "E step." << endl;
        // construct Q.
        const int Q = 100;
        vector<double> A(Q);
        for (int i = 0; i < Q; ++i) {
            A[i] = ((-1.) * i + (1.) * (Q - 1 - i)) / (Q - 1) * 3.;
        }

        // compute h_iq
        vector<vector<double> > h(examinee_count, vector<double>(Q));
        for (int i = 0; i < examinee_count; ++i) {
            double sum = 0;
            for (int q = 0; q < Q; ++q) {
                h[i][q] = model.response_probability(A[q], e_resp[i]) * normal_distribution(A[q]);
                sum += h[i][q];
            }
            for (int q = 0; q < Q; ++q) {
                h[i][q] /= sum;
            }
        }

        // compute r_qj, s_qj
        vector<vector<double> > r(Q, vector<double>(item_count)), s(Q, vector<double>(item_count));
        for (int q = 0; q < Q; ++q) {
            for (int it = 0; it < responses.size(); ++it) {
                const int i = responses[it].examinee;
                const int j = responses[it].item;
                if (responses[it].response) {
                    // positive response
                    r[q][j] += h[i][q];
                } else {
                    // negative response
                    s[q][j] += h[i][q];
                }
            }
        }

        // update model
        // by logistic regression
        cerr << "M step." << endl;
        double new_likelihood = 0;
        for (int j = 0; j < item_count; ++j) {
            cerr << "maximize item #" << j << endl;
            double &difficulty = model.difficulty[j], &discrimination = model.discrimination[j];
            const double alpha = 0.001;
            double last_lf = 0;
            for (int q = 0; q < Q; ++q) {
                const double positive_prob = model.probability_base(A[q], difficulty, discrimination);
                last_lf += r[q][j] * log(positive_prob) + s[q][j] * log(1 - positive_prob);
            }
            while (true) {
                cerr << "diff = " << difficulty << ", discrimination = " << discrimination << ", last_lf = " << last_lf << endl;
                double grad_diff = 0, grad_disc = 0;
                for (int q = 0; q < Q; ++q) {
                    const double positive_prob = model.probability_base(A[q], difficulty, discrimination);
                    const double D = 1.7;
                    grad_diff += (r[q][j] * (1. - positive_prob) + s[q][j] * (-positive_prob)) * (-D * discrimination);
                    grad_disc += (r[q][j] * (1. - positive_prob) + s[q][j] * (-positive_prob)) * (D * (A[q] - difficulty));
                }
                double n_diff = difficulty + alpha * grad_diff;
                double n_disc = discrimination + alpha * grad_disc;
                double lf = 0;
                for (int q = 0; q < Q; ++q) {
                    const double positive_prob = model.probability_base(A[q], n_diff, n_disc);
                    lf += r[q][j] * log(positive_prob) + s[q][j] * log(1 - positive_prob);
                }
                if (lf > last_lf + 1e-7) {
                    difficulty = n_diff;
                    discrimination = n_disc;
                    last_lf = lf;
                } else {
                    cerr << "n_diff = " << n_diff << ", n_disc = " << n_disc << ", lf = " << lf << endl;
                    break;
                }
            }
            new_likelihood += last_lf;
        }
        if (new_likelihood < likelihood + 1e-5) {
            break;
        } else {
            likelihood = new_likelihood;
        }
    }
    return model;
}

int main() {
    const int item = 10, examinee = 20;
    int ans[examinee][item] = {{1,1,1,1,1,1,1,1,0,0},
                               {1,1,1,1,1,1,0,1,1,0},
                               {1,1,1,1,1,1,0,1,0,0},
                               {1,1,1,1,1,1,0,0,0,0},
                               {1,1,1,0,1,1,0,1,0,0},
                               {1,1,1,1,1,1,0,0,0,0},
                               {1,1,1,1,1,1,0,0,0,0},
                               {1,1,1,1,0,1,0,0,0,1},
                               {1,0,1,1,0,1,0,1,0,0},
                               {1,1,1,0,1,1,0,0,0,0},
                               {1,1,1,0,1,1,0,0,0,0},
                               {1,1,1,1,0,1,0,0,0,0},
                               {1,1,1,0,1,1,0,0,0,0},
                               {1,1,1,0,1,1,0,0,0,0},
                               {1,1,1,1,0,0,0,0,0,0},
                               {1,1,1,0,0,1,0,0,0,0},
                               {1,1,1,0,1,0,0,0,0,0},
                               {1,1,1,0,0,1,0,0,0,0},
                               {1,1,1,0,0,1,0,0,0,0},
                               {1,1,1,0,0,1,0,0,0,0}};

    vector<Response> sample;
    for (int i = 0; i < examinee; ++i) {
        for (int j = 0; j < item; ++j) {
            sample.push_back((Response){i, j, ans[i][j]});
        }
    }

    Model em_model = estimate(sample, item, examinee);
    return 0;
}
