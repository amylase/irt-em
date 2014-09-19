#include <vector>
#include <cmath>

#include <iostream>

using namespace std;

const double inf = 1e300;
const double eps = 1e-7;

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

struct TestResults {
    int examinee_count;
    int item_count;
    vector<Response> responses;

    TestResults (int e, int i, const vector<Response>& r): examinee_count(e), item_count(i), responses(r) {}

    vector<vector<Response> > get_examinee_resp_list () const {
        vector<vector<Response> > e_resp(examinee_count, vector<Response>());
        for (int i = 0; i < responses.size(); ++i) {
            e_resp[responses[i].examinee].push_back(responses[i]);
        }
        return e_resp;
    }
};

struct Model {
    const int item_count;
    vector<double> discrimination, difficulty;
    Model (int ic): item_count(ic), discrimination(item_count, 1), difficulty(item_count, 0) {}

    double probability_base(double ability, double diff, double disc) const {
        // 2PLM IRT
        const double D = 1.7;
        return sigmoid(D * disc * (ability - diff));
    }

    double response_probability (double ability, const Response& response) const {
        int item_index = response.item;
        const double base = probability_base(ability, difficulty[item_index], discrimination[item_index]);
        return base * (response.response) + (1. - base) * (1 - response.response);
    }
    
    double response_probability (double ability, const vector<Response>& response) const {
        double ans = 1.;
        for (int i = 0; i < response.size(); ++i) {
            ans *= response_probability(ability, response[i]);
        }
        return ans;
    }

    vector<double> estimate_ability (const TestResults& results) const {
        vector<double> abilities(results.examinee_count, 0.);
        vector<vector<Response> > e_resp = results.get_examinee_resp_list();
        for (int i = 0; i < results.examinee_count; ++i) {
            // estimate by logistic regression using gradient descent
            double &ability = abilities[i];
            double ll = 0;
            for (int x = 0; x < e_resp[i].size(); ++x) {
                ll += log(response_probability(ability, e_resp[i][x]));
            }
            while (true) {
                const double alpha = 0.001, D = 1.7;
                double grad_ability = 0;
                for (int x = 0; x < e_resp[i].size(); ++x) {
                    const int j = e_resp[i][x].item;
                    grad_ability += D * discrimination[j] * (e_resp[i][x].response - probability_base(ability, difficulty[j], discrimination[j]));
                }
                const double new_ability = ability + alpha * grad_ability;
                double n_ll = 0;
                for (int x = 0; x < e_resp[i].size(); ++x) {
                    n_ll += log(response_probability(new_ability, e_resp[i][x]));
                }   
                if (n_ll > ll + eps) {
                    ability = new_ability;
                    ll = n_ll;
                } else {
                    break;
                }
            }
        }
        return abilities;
    }

    void debug(const TestResults& results) const {
        cerr << "item parameters." << endl;
        for (int j = 0; j < item_count; ++j) {
            cerr << "#" << j << " : difficulty = " << difficulty[j] << ", discrimination = " << discrimination[j] << endl;
        }
        vector<double> abilities = estimate_ability(results);
        cerr << "examinee abilities." << endl;
        for (int i = 0; i < results.examinee_count; ++i) {
            cerr << "#" << i << " : ability = " << abilities[i] << endl;
        }
        return;
    }
};

Model estimate (const TestResults& results) {
    const vector<Response> responses = results.responses;
    const int item_count = results.item_count;
    const int examinee_count = results.examinee_count;

    Model model(item_count);
    vector<vector<Response> > e_resp = results.get_examinee_resp_list();

    cerr << "EM algorithm start." << endl;

    double likelihood = -inf;
    while (true) {
        cerr << "Loop start. likelihood = " << likelihood << endl;
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
        double new_likelihood = 0;
        Model old_model = model;
        for (int j = 0; j < item_count; ++j) {
            double &difficulty = model.difficulty[j], &discrimination = model.discrimination[j];
            const double alpha = 0.001;
            double last_lf = 0;
            for (int q = 0; q < Q; ++q) {
                const double positive_prob = model.probability_base(A[q], difficulty, discrimination);
                last_lf += r[q][j] * log(positive_prob) + s[q][j] * log(1 - positive_prob);
            }
            while (true) {
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
                if (lf > last_lf + eps) {
                    difficulty = n_diff;
                    discrimination = n_disc;
                    last_lf = lf;
                } else {
                    break;
                }
            }
            new_likelihood += last_lf;
        }
        if (new_likelihood < likelihood + eps) {
            return old_model;
        } else {
            likelihood = new_likelihood;
        }
    }
    return model;
}

int main() {
    int response_count, examinee_count, item_count;
    cin >> response_count >> examinee_count >> item_count;

    vector<Response> responses;
    for (int i = 0; i < response_count; ++i) {
        int examinee, item, response;
        cin >> examinee >> item >> response;
        responses.push_back((Response){examinee, item, response});
    }

    TestResults results(examinee_count, item_count, responses);
    Model em_model = estimate(results);
    em_model.debug(results);
    return 0;
}
