import numpy as np


def get_domain_stat(client_stats, client_datasets, domain_names):
    """
    Aggregate client statistics to domain statistics.
    :param client_stats:
    :param client_datasets:
    :param domain_names:
    :return:
    """
    num_clients_per_domain = len(client_datasets) // len(domain_names)

    domain_stats = {}

    for domain_name in domain_names:
        num_samples = 0
        num_correct = 0

        for i in range(num_clients_per_domain):

            num_samples += len(client_datasets[f"{domain_name}_{i}"])
            num_correct += round(len(client_datasets[f"{domain_name}_{i}"]) * client_stats[f"{domain_name}_{i}"])

        domain_stats[domain_name] = num_correct / num_samples

    return domain_stats

