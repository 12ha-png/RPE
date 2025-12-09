import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
from typing import List, Tuple, Dict, Set, Any
import random
import time
from collections import Counter


class RPE_EntityMatcher:
    def __init__(self):
        self.probed_pairs = set()
        self.probed_labels = {}
        self.feature_vectors = {}
        self.brand_patterns = set()

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ""

        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s\.\-\+\®\©\™]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_price(self, price_str: str) -> float:
        if pd.isna(price_str) or price_str == '':
            return 0.0

        price_str = str(price_str)
        matches = re.findall(r'\$?(\d+\.?\d*)', price_str)
        if matches:
            return float(matches[0])
        return 0.0

    def similarity(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def learn_brands_from_data(self, abt_df: pd.DataFrame, buy_df: pd.DataFrame):
        brand_candidates = Counter()

        for df in [abt_df, buy_df]:
            for name in df['name'].dropna():
                if isinstance(name, str):
                    words = re.findall(r'\b[A-Z][a-z]+\b', name)
                    for word in words:
                        if len(word) > 2:
                            brand_candidates[word.lower()] += 1

                    name_lower = name.lower()
                    common_brand_patterns = [
                        r'\b([a-z]+)\s+(router|switch|adapter|cable|printer|monitor|keyboard|mouse)\b',
                        r'\b(router|switch|adapter|cable|printer|monitor|keyboard|mouse)\s+([a-z]+)\b',
                        r'\b([a-z]+)\s+(wireless|ethernet|network|usb|bluetooth)\b',
                    ]

                    for pattern in common_brand_patterns:
                        matches = re.findall(pattern, name_lower)
                        for match in matches:
                            for group in match:
                                if group and len(group) > 2:
                                    brand_candidates[group] += 1

        min_frequency = 2
        self.brand_patterns = {brand for brand, count in brand_candidates.items()
                               if count >= min_frequency and len(brand) > 2}

        print(f"从数据中学习到 {len(self.brand_patterns)} 个品牌模式")
        if self.brand_patterns:
            print(f"前10个品牌模式: {list(self.brand_patterns)[:10]}")

    def extract_brand(self, row: pd.Series) -> str:
        name = str(row['name']).lower()

        for brand in self.brand_patterns:
            if brand in name:
                return brand

        name_words = name.split()
        if name_words:
            first_word = name_words[0]
            if len(first_word) > 2 and not first_word.isdigit():
                common_suffixes = ['tech', 'sys', 'net', 'com', 'pro', 'max', 'plus']
                has_suffix = any(suffix in first_word for suffix in common_suffixes)

                product_terms = ['wireless', 'ethernet', 'network', 'usb', 'adapter',
                                 'cable', 'switch', 'router', 'printer', 'monitor']
                is_product_term = any(term in first_word for term in product_terms)

                if not is_product_term and not has_suffix:
                    return first_word

        return ""

    def create_feature_vector(self, abt_row: pd.Series, buy_row: pd.Series) -> Tuple[float, ...]:
        abt_name = self.preprocess_text(abt_row['name'])
        buy_name = self.preprocess_text(buy_row['name'])
        abt_desc = self.preprocess_text(abt_row.get('description', ''))
        buy_desc = self.preprocess_text(buy_row.get('description', ''))

        name_seq_sim = SequenceMatcher(None, abt_name, buy_name).ratio()

        abt_name_tokens = set(abt_name.split())
        buy_name_tokens = set(buy_name.split())
        name_jaccard_sim = self.similarity(abt_name_tokens, buy_name_tokens)

        desc_sim = SequenceMatcher(None, abt_desc, buy_desc).ratio()

        abt_brand = self.extract_brand(abt_row)
        buy_brand = self.extract_brand(buy_row)
        if abt_brand and buy_brand:
            brand_sim = SequenceMatcher(None, abt_brand, buy_brand).ratio()
        else:
            brand_sim = 0.0

        abt_price = self.extract_price(abt_row.get('price', ''))
        buy_price = self.extract_price(buy_row.get('price', ''))
        if abt_price > 0 and buy_price > 0:
            price_sim = 1 - min(abs(abt_price - buy_price) / max(abt_price, buy_price), 1.0)
        else:
            price_sim = 0.0

        name_combined_sim = (name_seq_sim + name_jaccard_sim) / 2

        return (name_combined_sim, desc_sim, brand_sim, price_sim)

    def dominates(self, point1: Tuple[float, ...], point2: Tuple[float, ...]) -> bool:

        all_greater_equal = all(p1 >= p2 for p1, p2 in zip(point1, point2))
        any_strictly_greater = any(p1 > p2 for p1, p2 in zip(point1, point2))
        return all_greater_equal and any_strictly_greater

    def blocking(self, abt_df: pd.DataFrame, buy_df: pd.DataFrame) -> List[Tuple[Any, Any]]:

        candidate_pairs = []


        self.learn_brands_from_data(abt_df, buy_df)


        abt_brands = {}
        for _, row in abt_df.iterrows():
            brand = self.extract_brand(row)
            if brand:
                if brand not in abt_brands:
                    abt_brands[brand] = []
                abt_brands[brand].append(row['id'])

        buy_brands = {}
        for _, row in buy_df.iterrows():
            brand = self.extract_brand(row)
            if brand:
                if brand not in buy_brands:
                    buy_brands[brand] = []
                buy_brands[brand].append(row['id'])


        common_brands = set(abt_brands.keys()).intersection(set(buy_brands.keys()))
        for brand in common_brands:
            for abt_id in abt_brands[brand]:
                for buy_id in buy_brands[brand]:
                    candidate_pairs.append((abt_id, buy_id))

        if len(candidate_pairs) < min(len(abt_df), len(buy_df)):
            print("品牌阻塞得到的候选对较少，尝试基于名称相似度的阻塞...")

            abt_names = {}
            for _, row in abt_df.iterrows():
                name = self.preprocess_text(row['name'])
                if name:
                    key_words = [word for word in name.split() if len(word) > 3]
                    for key_word in key_words[:2]:
                        if key_word not in abt_names:
                            abt_names[key_word] = []
                        abt_names[key_word].append(row['id'])

            buy_names = {}
            for _, row in buy_df.iterrows():
                name = self.preprocess_text(row['name'])
                if name:
                    key_words = [word for word in name.split() if len(word) > 3]
                    for key_word in key_words[:2]:
                        if key_word not in buy_names:
                            buy_names[key_word] = []
                        buy_names[key_word].append(row['id'])

            common_words = set(abt_names.keys()).intersection(set(buy_names.keys()))
            for word in common_words:
                for abt_id in abt_names[word]:
                    for buy_id in buy_names[word]:
                        pair = (abt_id, buy_id)
                        if pair not in candidate_pairs:
                            candidate_pairs.append(pair)

        print(f"阻塞后候选对数量: {len(candidate_pairs)}")
        return candidate_pairs

    def rpe_algorithm(self, abt_df: pd.DataFrame, buy_df: pd.DataFrame,
                      true_mapping: Set[Tuple], max_probes: int = None) -> Tuple[Any, Set, int, Dict]:


        candidate_pairs = self.blocking(abt_df, buy_df)

        if not candidate_pairs:
            print("警告：阻塞后没有候选对，使用全连接")
            candidate_pairs = []
            for _, abt_row in abt_df.iterrows():
                for _, buy_row in buy_df.iterrows():
                    candidate_pairs.append((abt_row['id'], buy_row['id']))

        abt_dict = {row['id']: row for _, row in abt_df.iterrows()}
        buy_dict = {row['id']: row for _, row in buy_df.iterrows()}

        print("计算特征向量...")
        for i, pair in enumerate(candidate_pairs):
            abt_id, buy_id = pair
            if abt_id in abt_dict and buy_id in buy_dict:
                self.feature_vectors[pair] = self.create_feature_vector(
                    abt_dict[abt_id], buy_dict[buy_id]
                )

            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1} 个候选对")

        P = set(candidate_pairs)  
        Z = set()
        if max_probes is None:
            max_probes = min(len(candidate_pairs), 1000)

        probe_count = 0

        print(f"开始RPE算法，最大探测数: {max_probes}")

        while P and probe_count < max_probes:
            p = random.choice(list(P))
            P.remove(p)


            abt_id, buy_id = p
            label = 1 if (abt_id, buy_id) in true_mapping else 0
            Z.add(p)
            self.probed_labels[p] = label
            probe_count += 1

            if p in self.feature_vectors:
                p_features = self.feature_vectors[p]

                if label == 1:
                    to_remove = set()
                    for q in P:
                        if (q in self.feature_vectors and
                                self.dominates(self.feature_vectors[q], p_features)):  # q支配p
                            to_remove.add(q)
                    P -= to_remove

                else:

                    to_remove = set()
                    for q in P:
                        if (q in self.feature_vectors and
                                self.dominates(p_features, self.feature_vectors[q])):  # p支配q
                            to_remove.add(q)
                    P -= to_remove

            if probe_count % 50 == 0:
                print(f"已完成 {probe_count} 次探测，剩余候选对: {len(P)}")

        print(f"RPE算法完成，总共探测 {probe_count} 个点")

        def classifier(pair):

            if pair in self.probed_labels:
                return self.probed_labels[pair]

            for probed_pair in Z:
                if (self.probed_labels[probed_pair] == 1 and
                        pair in self.feature_vectors and
                        probed_pair in self.feature_vectors and
                        self.dominates(self.feature_vectors[pair], self.feature_vectors[probed_pair])):
                    return 1
            return 0

        return classifier, Z, probe_count, self.feature_vectors

    def evaluate_performance(self, classifier, candidate_pairs: List[Tuple],
                             true_mapping: Set[Tuple]) -> Dict[str, Any]:

        predictions = {}
        true_labels = {}


        for i, pair in enumerate(candidate_pairs):
            true_labels[pair] = 1 if pair in true_mapping else 0
            predictions[pair] = classifier(pair)

            if (i + 1) % 1000 == 0:
                print(f"已评估 {i + 1} 个候选对")

        tp = fp = fn = tn = 0
        for pair in candidate_pairs:
            true = true_labels[pair]
            pred = predictions[pair]

            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / len(candidate_pairs) if candidate_pairs else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        probe_count = len(self.probed_labels)
        total_pairs = len(candidate_pairs)
        cost_saving = (1 - probe_count / total_pairs) * 100 if total_pairs > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'probe_count': probe_count,
            'total_pairs': total_pairs,
            'cost_saving': cost_saving,
            'predictions': predictions,
            'true_labels': true_labels
        }

    def analyze_dominance_relationships(self, candidate_pairs: List[Tuple], sample_size: int = 5):


        if not self.feature_vectors:
            print("没有特征向量数据")
            return

        sample_pairs = random.sample(candidate_pairs, min(sample_size, len(candidate_pairs)))

        for i, pair1 in enumerate(sample_pairs):
            if pair1 not in self.feature_vectors:
                continue

            features1 = self.feature_vectors[pair1]
            print(f"\n点 {pair1}: 特征={features1}")

            dominating_count = 0
            dominated_count = 0

            for pair2 in sample_pairs:
                if pair1 == pair2 or pair2 not in self.feature_vectors:
                    continue

                features2 = self.feature_vectors[pair2]

                if self.dominates(features1, features2):
                    dominated_count += 1
                    print(f"  支配点 {pair2} (特征={features2})")

                if self.dominates(features2, features1):
                    dominating_count += 1
                    print(f"  被点 {pair2} 支配 (特征={features2})")

            print(f"  支配 {dominated_count} 个点，被 {dominating_count} 个点支配")


def load_and_preprocess_data(abt_path: str, buy_path: str, mapping_path: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, Set[Tuple]]:

    abt_df = pd.read_csv(abt_path)
    buy_df = pd.read_csv(buy_path)
    mapping_df = pd.read_csv(mapping_path)

    true_mapping = set()
    for _, row in mapping_df.iterrows():
        true_mapping.add((row['idAbt'], row['idBuy']))

    print("=== 数据集统计 ===")
    print(f"Abt 记录数: {len(abt_df)}")
    print(f"Buy 记录数: {len(buy_df)}")
    print(f"真实匹配数: {len(true_mapping)}")
    print(f"理论候选对总数: {len(abt_df) * len(buy_df)}")

    return abt_df, buy_df, true_mapping


def run_comprehensive_experiment():

    abt_df, buy_df, true_mapping = load_and_preprocess_data(
        'Abt_1.csv',
        'Buy.csv',
        'abt_buy_perfectMapping.csv'
    )

    matcher = RPE_EntityMatcher()

    start_time = time.time()

    max_probes = min(200, len(abt_df) * len(buy_df) // 10)
    classifier, Z, probe_count, feature_vectors = matcher.rpe_algorithm(
        abt_df, buy_df, true_mapping, max_probes=max_probes
    )

    end_time = time.time()

    candidate_pairs = matcher.blocking(abt_df, buy_df)
    if not candidate_pairs:
        candidate_pairs = [(abt_row['id'], buy_row['id'])
                           for _, abt_row in abt_df.iterrows()
                           for _, buy_row in buy_df.iterrows()]


    results = matcher.evaluate_performance(classifier, candidate_pairs, true_mapping)

    print("\n=== RPE算法实验结果 ===")
    print(f"运行时间: {end_time - start_time:.2f} 秒")
    print(f"探测数量 (成本): {results['probe_count']}")
    print(f"评估候选对数量: {results['total_pairs']}")
    print(f"成本节约: {results['cost_saving']:.2f}%")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1_score']:.4f}")
    print(f"真正例 (TP): {results['true_positives']}")
    print(f"假正例 (FP): {results['false_positives']}")
    print(f"假反例 (FN): {results['false_negatives']}")
    print(f"真反例 (TN): {results['true_negatives']}")

    optimal_probes = len(true_mapping)
    theoretical_optimal_cost = optimal_probes
    actual_cost = results['probe_count']

    print(f"\n=== 成本分析 ===")
    print(f"理论最优成本 (探测所有真实匹配): {theoretical_optimal_cost}")
    print(f"实际成本: {actual_cost}")
    print(f"成本效率: {(theoretical_optimal_cost / actual_cost * 100):.2f}%" if actual_cost > 0 else "N/A")


    matcher.analyze_dominance_relationships(candidate_pairs, sample_size=5)

    return results, matcher


if __name__ == "__main__":
    results, matcher = run_comprehensive_experiment()