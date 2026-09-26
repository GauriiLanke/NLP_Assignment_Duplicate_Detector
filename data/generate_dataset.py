"""
Dataset Generation Script
Creates a diverse, realistic dataset of student assignment pairs across 6 CS subjects:
- Artificial Intelligence & Machine Learning
- Database Management Systems (DBMS)
- Operating Systems (OS)
- Computer Networks (CN)
- Cloud Computing
- Cyber Security

Generates 600+ labeled pairs:
- Label 2: Duplicate (Exact copy or near-identical)
- Label 1: Similar (Paraphrased, rewritten, reordered, structural modifications)
- Label 0: Different (Completely different assignments or unrelated topics)
"""

import os
import random
import pandas as pd

random.seed(42)

BASE_TOPICS = {
    "AI_ML": [
        "Machine learning is a subset of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed. Common algorithms include decision trees, neural networks, and support vector machines.",
        "Supervised learning algorithms are trained using labeled datasets where the desired output is known. Unsupervised learning models discover hidden patterns or groupings in unlabeled datasets, such as k-means clustering.",
        "Convolutional Neural Networks are deep neural network architectures specifically designed for computer vision and image processing tasks. They use convolutional layers to extract spatial hierarchies of features from images.",
        "Natural language processing allows computers to interpret, analyze, and manipulate human language using statistical models, rule-based heuristics, and deep learning transformers like BERT and GPT.",
        "Reinforcement learning is an area of machine learning concerning how intelligent agents ought to take actions in an environment in order to maximize cumulative reward through exploration and exploitation."
    ],
    "DBMS": [
        "A relational database management system organizes data into tables consisting of rows and columns with primary keys and foreign keys enforcing referential integrity.",
        "ACID properties ensure database transaction reliability: Atomicity guarantees all-or-nothing execution, Consistency enforces constraints, Isolation separates concurrent transactions, and Durability ensures committed updates persist.",
        "Database normalization is the systematic process of organizing fields and tables to minimize data redundancy and dependency anomalies, progressing through 1NF, 2NF, 3NF, and BCNF.",
        "SQL indexing creates data structures such as B-trees or hash tables on table columns to drastically speed up query retrieval performance at the cost of additional storage and write overhead.",
        "NoSQL databases provide flexible schema designs and horizontal scalability for distributed data models, including document stores like MongoDB, key-value stores like Redis, and graph databases like Neo4j."
    ],
    "OS": [
        "An operating system acts as an intermediary between computer hardware and user applications, providing process management, memory management, file systems, and device control.",
        "Process scheduling algorithms determine the order in which execution threads receive CPU time, including First-Come-First-Served, Shortest Job Next, Priority Scheduling, and Round Robin.",
        "Deadlock occurs in concurrent programming when two or more processes are permanently blocked because each is waiting for a resource held by another process in a circular wait condition.",
        "Virtual memory uses paging and swapping mechanisms to combine physical RAM with secondary disk storage, allowing programs larger than physical memory to execute smoothly.",
        "Semaphores and mutex locks are synchronization primitives designed to prevent race conditions and enforce mutual exclusion in multithreaded concurrent systems."
    ],
    "NETWORKS": [
        "The OSI reference model defines network communication through seven conceptual layers: Physical, Data Link, Network, Transport, Session, Presentation, and Application.",
        "Transmission Control Protocol provides connection-oriented, reliable, and ordered delivery of byte streams between networked hosts through three-way handshakes and flow control.",
        "User Datagram Protocol is a connectionless transport protocol prioritizing low latency over reliability, frequently used in real-time video streaming, DNS lookups, and gaming.",
        "Routing algorithms such as Dijkstra's link-state and Bellman-Ford distance-vector establish optimum packet paths across interconnected autonomous system networks.",
        "Domain Name System translates human-friendly domain names like example.com into numerical IP addresses required for packet routing across the internet."
    ],
    "CLOUD": [
        "Cloud computing delivers computing services including servers, storage, databases, networking, and software over the internet with on-demand elasticity and pay-as-you-go pricing.",
        "Cloud service models are categorized into Infrastructure as a Service, Platform as a Service, and Software as a Service depending on the degree of customer management.",
        "Containerization packages application code together with its runtime dependencies into portable containers orchestrated at scale using tools like Docker and Kubernetes.",
        "Serverless computing allows developers to deploy event-driven functions without provisioning or managing underlying virtual servers, scaling automatically to zero when idle.",
        "Microservices architecture structures an enterprise application as a collection of loosely coupled, independently deployable services communicating via REST APIs or message brokers."
    ],
    "SECURITY": [
        "Public-key cryptography utilizes asymmetric key pairs consisting of a public key for encryption and a private mathematically linked key for decryption.",
        "A firewall monitors and filters incoming and outgoing network traffic based on predefined security rules to block unauthorized access and malicious threats.",
        "SQL injection occurs when untrusted user input is directly concatenated into SQL database queries, allowing attackers to manipulate queries and extract confidential data.",
        "Multi-factor authentication strengthens access control by requiring two or more independent credentials such as passwords, authenticator tokens, or biometrics.",
        "Denial of Service and Distributed Denial of Service attacks attempt to disrupt the normal traffic of a targeted server by overwhelming it with a flood of malicious traffic."
    ]
}

PARAPHRASE_PATTERNS = [
    ("Machine learning is a subset of artificial intelligence", "ML is a specialized branch within AI"),
    ("learn from data and improve their accuracy", "extract patterns from training data to boost precision"),
    ("Common algorithms include", "Popular computational models encompass"),
    ("Supervised learning algorithms are trained using labeled datasets", "Supervised approaches utilize annotated data sets with known ground truth"),
    ("Unsupervised learning models discover hidden patterns", "Unsupervised methods detect latent groupings"),
    ("specifically designed for computer vision", "tailored for digital image recognition and visual analysis"),
    ("relational database management system organizes data into tables", "RDBMS structures records into relational tabular schemas"),
    ("ACID properties ensure database transaction reliability", "The ACID paradigm guarantees trustworthy transaction processing"),
    ("operating system acts as an intermediary", "The OS serves as a software bridge"),
    ("Deadlock occurs in concurrent programming", "A deadlock situation arises during multithreading"),
    ("OSI reference model defines network communication", "The 7-layer OSI model formalizes telecommunication protocols"),
    ("Cloud computing delivers computing services", "Cloud infrastructure provides on-demand computational capabilities")
]


def paraphrase_text(text: str) -> str:
    """Applies realistic paraphrasing and lexical substitution to a text."""
    p_text = text
    for orig, rep in PARAPHRASE_PATTERNS:
        if orig in p_text:
            p_text = p_text.replace(orig, rep)

    # Synonym replacements
    replacements = {
        "algorithms": "models",
        "computers": "digital machines",
        "programs": "software applications",
        "tasks": "operations",
        "reliable": "dependable",
        "storage": "memory capacity",
        "performance": "efficiency",
        "services": "utilities",
        "developers": "software engineers",
        "attacks": "cyber intrusions",
        "confidential": "sensitive private"
    }
    words = p_text.split()
    new_words = [replacements.get(w.lower().strip(".,"), w) for w in words]
    return " ".join(new_words)


def generate_dataset(num_pairs_per_class: int = 220) -> pd.DataFrame:
    """
    Generates balanced dataset:
    220 Duplicate pairs (Label: Duplicate)
    220 Similar pairs (Label: Similar)
    220 Different pairs (Label: Different)
    Total: 660 labeled assignment pairs
    """
    all_texts = []
    for cat_list in BASE_TOPICS.values():
        all_texts.extend(cat_list)

    pairs = []
    pair_id = 1

    # 1. DUPLICATE PAIRS (Label: Duplicate -> 2)
    for _ in range(num_pairs_per_class):
        base = random.choice(all_texts)
        if random.random() < 0.6:
            # Exact copy
            t1, t2 = base, base
        else:
            # Minor typo or small edit
            words = base.split()
            if len(words) > 5:
                idx = random.randint(0, len(words) - 1)
                words2 = list(words)
                words2[idx] = words2[idx] + " "
                t1 = base
                t2 = " ".join(words2)
            else:
                t1, t2 = base, base
        pairs.append({
            "pair_id": pair_id,
            "assignment1": t1,
            "assignment2": t2,
            "label_name": "Duplicate",
            "label": 2
        })
        pair_id += 1

    # 2. SIMILAR PAIRS (Label: Similar -> 1)
    for _ in range(num_pairs_per_class):
        base = random.choice(all_texts)
        # Create paraphrased and expanded version
        para = paraphrase_text(base)
        # Add slight variation or commentary
        expansions = [
            " In practical applications, this provides robust capabilities.",
            " This concept is fundamental to modern computing systems.",
            " Industry adoption of this method has increased drastically.",
            " Extensive research continues to enhance its operational performance."
        ]
        t1 = base
        t2 = para + random.choice(expansions)
        pairs.append({
            "pair_id": pair_id,
            "assignment1": t1,
            "assignment2": t2,
            "label_name": "Similar",
            "label": 1
        })
        pair_id += 1

    # 3. DIFFERENT PAIRS (Label: Different -> 0)
    for _ in range(num_pairs_per_class):
        cats = list(BASE_TOPICS.keys())
        c1, c2 = random.sample(cats, 2)
        t1 = random.choice(BASE_TOPICS[c1])
        t2 = random.choice(BASE_TOPICS[c2])
        pairs.append({
            "pair_id": pair_id,
            "assignment1": t1,
            "assignment2": t2,
            "label_name": "Different",
            "label": 0
        })
        pair_id += 1

    df = pd.DataFrame(pairs)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "assignments_dataset.csv")
    print("Generating synthetic assignment dataset...")
    df = generate_dataset(num_pairs_per_class=220)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Dataset successfully created at: {csv_path}")
    print(f"Total pairs: {len(df)}")
    print(df["label_name"].value_counts())
