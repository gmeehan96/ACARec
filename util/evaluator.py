import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def hit_rate(hits, origin):
        recall_list = [int(hits[user] > 0) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    performance = [[] for _ in N]
    for i, n in enumerate(N):
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print(
                f"ground-truth set size: {len(origin)}, predicted set size: {len(predicted)}"
            )
            print("The Lengths of ground-truth set and predicted set do not match!")
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_rate(origin, hits)
        indicators.append("Hit Rate:" + str(hr) + "\n")

        recall = Metric.recall(hits, origin)
        indicators.append("Recall:" + str(recall) + "\n")

        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append("NDCG:" + str(NDCG) + "\n")
        measure.append("Top " + str(n) + "\n")
        measure += indicators

        performance[i].append(hr)
        performance[i].append(recall)
        performance[i].append(NDCG)
    return measure, performance


