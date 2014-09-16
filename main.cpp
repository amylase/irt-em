#include <vector>
#include <cmath>

using namespace std;

struct Response {
    int examinee;
    int item;
    int response;
};

struct Model {
    const int item_count;
    vector<double> discrimination, difficulty;
    Model (int ic): item_count(ic), discrimination(item_count), difficulty(item_count) {}
    double probability_base(int item, double ability) {
        // 2PLM IRT
        const double D = 1.7;
        return 1. / (1. + exp(-D * discrimination[item] * (ability - difficulty[item]))); 
    }

    double response_probability (double ability, const Response& response) {
        const double base = probability_base(response.item, ability);
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

double normal_distribution(double x) {
    return exp(-0.5 * pow(x, 2));
}

Model estimate (const vector<Response>& responses, int item_count, int examinee_count) {
    Model model(item_count);
    vector<vector<Response> > e_resp(examinee_count, vector<Response>());
    for (int i = 0; i < responses.size(); ++i) {
        e_resp[responses[i].examinee].push_back(responses[i]);
    }

    double likelihood = -1e300;
    while (true) {
        // construct Q.
        const int Q = 100;
        vector<double> A(Q);
        for (int i = 0; i < Q; ++i) {
            A[i] = ((-1.) * i + (1.) * (Q - 1 - i)) / (Q - 1);
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
        
    }
    return model;
}

int main() {

    return 0;
}
