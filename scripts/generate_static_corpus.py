"""
Static Benchmark Corpus Generator (120 Document Pairs across 5 Domains).
Paper: "Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks"
Generates 120 rich, diverse document pairs (24 per domain) with pre-structured source, suspect,
and adversarial candidate paraphrases, fully compatible with external population (e.g. Claude, GPT).
"""

import json
import os
from typing import List, Dict

def create_120_benchmark_pairs() -> List[Dict]:
    pairs = []
    
    # -------------------------------------------------------------------------
    # DOMAIN 1: Academic / PlagBench (24 pairs)
    # -------------------------------------------------------------------------
    academic_topics = [
        ("Representation Learning",
         "Supervised contrastive learning optimizes the representation space by pulling together normalized embeddings from the same class while pushing apart embeddings from different classes. The loss function extends the traditional self-supervised InfoNCE framework to accommodate multiple positive pairs, enabling robust multi-class feature clustering and mitigating class collapse.",
         "Supervised contrastive learning structures the latent space by clustering normalized embeddings belonging to identical classes and separating representations from distinct classes. This objective generalizes the conventional unsupervised InfoNCE loss to handle multiple positive instances, which facilitates stable feature discrimination and avoids representation collapse.",
         "By grouping unit-length vectors from identical classes and distancing disparate categories, supervised contrastive learning shapes the feature geometry. This method broadens the unsupervised InfoNCE loss to manage numerous positive samples, preventing feature degeneration while promoting class separation."),
        
        ("Transformer Complexity",
         "Transformer architectures rely heavily on multi-head self-attention mechanisms to compute pairwise relationships between all tokens in a sequence. Although effective for short sequences, the quadratic computational complexity with respect to sequence length imposes severe memory bottlenecks during document-level representation learning.",
         "Transformers depend on multi-head self-attention layers to model token-to-token dependencies across the input sequence. While highly expressive for brief texts, the quadratic time and memory scaling relative to context length introduces substantial computational bottlenecks when analyzing long-form documents.",
         "Token-to-token dependencies across input sequences are modeled in transformers via parallelized attention heads. Even though powerful for concise passages, quadratic scaling in processing and memory creates major resource limits when handling voluminous contexts."),

        ("Diffusion Probabilistic Models",
         "Denoising diffusion probabilistic models generate high-fidelity samples by iteratively reversing a parameterized Markovian forward process that systematically corrupts data with Gaussian noise. Training optimizes a variational bound on the data likelihood parameterized as an objective predicting the added noise vectors at each timestep.",
         "Diffusion models synthesize high-quality images by progressively inverting a forward Markov chain that incrementally perturbs input images with Gaussian noise. The neural network optimizes a variational lower bound by estimating the exact noise component introduced at each discrete timestep.",
         "Generative diffusion models reconstruct high-fidelity data by systematically undoing a Markovian corruption process governed by Gaussian perturbations. The network minimizes a variational likelihood objective by estimating the injected noise vector across successive time steps."),

        ("Graph Neural Networks",
         "Message passing neural networks update node representations by iteratively aggregating localized feature vectors from adjacent graph vertices through non-linear permutation-invariant permutation pooling functions followed by gated recurrent update functions.",
         "Graph message-passing architectures refine vertex embeddings by recursively gathering neighbor node features using permutation-invariant aggregation functions coupled with non-linear neural transformation layers.",
         "Node embeddings in graph neural networks are iteratively updated by collecting feature vectors from neighboring vertices via permutation-invariant pooling and applying parameterized non-linear state transitions."),

        ("Retrieval-Augmented Generation",
         "Retrieval-augmented language models integrate external parametric knowledge stores by conditioning generation on relevant document chunks retrieved via dense vector similarity indexes during prompt construction.",
         "RAG systems enhance generative models with external memory by appending relevant context passages identified through dense semantic search algorithms to the input prompt.",
         "To enrich language model responses with external knowledge, RAG frameworks retrieve contextual passages from dense vector stores and concatenate them into the generation prompt."),

        ("Reinforcement Learning from Human Feedback",
         "RLHF aligns language model generation with human intent by fitting a reward model over pairwise preference annotations and optimizing the base policy using Proximal Policy Optimization with a KL-divergence penalty constraint.",
         "Human feedback alignment trains a scalar reward predictor on paired preference comparisons, subsequently updating the generative policy using PPO while enforcing a KL regularization term to preserve fluency.",
         "Aligning language models with human preferences involves training a reward predictor from pairwise comparison datasets, followed by PPO policy optimization constrained by a KL divergence barrier to maintain coherence."),

        ("Low-Rank Adaptation (LoRA)",
         "Low-rank adaptation freezes the pre-trained model weights and injects trainable rank decomposition matrices into each transformer layer, drastically reducing the number of trainable parameters for downstream task adaptation.",
         "LoRA enables parameter-efficient fine-tuning by keeping base model parameters fixed while optimizing low-rank matrix pairs added to the transformer self-attention projections.",
         "Parameter-efficient adaptation with LoRA keeps underlying model weights frozen and optimizes low-rank factorized matrices across attention projections, substantially lowering compute requirements."),

        ("Quantization & Model Compression",
         "Post-training quantization compresses deep neural networks by mapping floating-point weights and activation tensors to lower-bit precision integers while minimizing the resulting reconstruction error via calibration datasets.",
         "Model compression through post-training quantization converts high-precision float weights to reduced-bit integer formats using calibration samples to minimize quantization noise.",
         "Quantizing neural network weights to low-bit integer representations reduces memory footprint, employing calibration data distributions to constrain precision loss."),

        ("Mixture of Experts (MoE)",
         "Sparse mixture of experts architectures route input tokens dynamically to a subset of specialized feed-forward sub-networks using a learnable gating mechanism, scaling parameter capacity without proportional computational overhead.",
         "MoE models increase effective model capacity without increasing per-token compute by using a routing router to dispatch tokens to a small fraction of specialized expert networks.",
         "Dynamic routing mechanisms in mixture-of-experts architectures direct token representations to select specialized sub-networks, scaling parameter count while keeping inference latency bounded."),

        ("Prompt Tuning",
         "Prefix and prompt tuning prepend trainable continuous virtual token vectors to transformer input keys and values, leaving the core language model parameters untouched during task fine-tuning.",
         "Continuous prompt tuning learns task-specific virtual prompt embeddings inserted into transformer attention layers while keeping all original network weights completely frozen.",
         "Task-specific adaptation via prompt tuning optimizes continuous virtual token vectors prepended to attention layers, entirely preserving base model parameters."),

        ("Multi-Modal Vision Transformers",
         "Vision transformers tokenize 2D image patches into linear sequence embeddings and process them with standard transformer encoders to learn contextual spatial representations for object recognition.",
         "ViT architectures partition input images into a grid of non-overlapping patches, projecting them into embedding vectors that are processed using standard self-attention blocks.",
         "By dividing 2D images into flattened sequential patches and projecting them into embedding space, vision transformers leverage self-attention to capture global spatial patterns."),

        ("Self-Supervised Masked Autoencoding",
         "Masked autoencoders reconstruct masked visual or linguistic patches from high-ratio corrupted sequences, compelling the underlying encoder to learn rich semantic representations of context.",
         "MAE models train neural representations by masking a large fraction of input tokens and training an encoder-decoder network to predict the omitted patches from visible context.",
         "By obscuring substantial proportions of input sequences and reconstructing omitted segments, masked autoencoders force encoders to capture deep contextual correlations."),

        ("Causal Inference in NLP",
         "Causal representation learning disentangles spurious correlations from true causal mechanisms in natural language text using structural causal models and invariant risk minimization principles.",
         "Causal NLP methods isolate invariant predictive features from confounding dataset biases by applying structural equation modeling and invariant risk frameworks.",
         "Separating true semantic mechanisms from dataset artifacts in NLP relies on causal DAGs and invariant risk minimization techniques."),

        ("Zero-Shot Generalization",
         "Instruction tuning transforms diverse NLP tasks into natural language prompts, allowing large language models to generalize zero-shot to unseen evaluation tasks without explicit parameter updates.",
         "Fine-tuning models on broad instruction datasets enables zero-shot execution of novel tasks by interpreting structured natural language instructions at inference time.",
         "Instruction-tuned foundation models execute previously unseen tasks in a zero-shot setting by conditioning on natural language task prompts without weight updates."),

        ("Contrastive Representation Learning",
         "InfoNCE contrastive learning maximizes mutual information between alternative augmented views of identical data instances while minimizing mutual information across distinct instances in latent space.",
         "Contrastive learning objectives maximize embedding agreement across augmented versions of the same sample while repelling representations of different samples.",
         "Mutual information between augmented views of the same data point is maximized in InfoNCE contrastive frameworks while separating negative samples."),

        ("Long-Context Attention Mechanisms",
         "Linear attention approximations replace the quadratic softmax kernel with kernel feature maps, allowing transformers to scale linearly with sequence length during document processing.",
         "Approximating softmax attention with decomposed kernel functions enables linear time and memory scaling for processing long-context text sequences.",
         "Kernel-based linear attention models bypass softmax quadratic bottlenecks, achieving linear computational complexity for long document comprehension."),

        ("Hallucination Mitigation",
         "Factual consistency evaluation in summarization computes atomic fact decomposition and checks entailment against source documents to identify and suppress model hallucinations.",
         "Detecting hallucinations in generated text decomposes summaries into independent claims and verifies whether each claim is entailed by the reference document.",
         "Factuality checking pipelines segment generated passages into discrete atomic propositions and evaluate bidirectional NLI entailment against reference sources."),

        ("Knowledge Distillation",
         "Knowledge distillation transfers dark knowledge from a large teacher model to a compact student network by minimizing the Kullback-Leibler divergence between their softened output probability distributions.",
         "Student-teacher distillation trains compact models by aligning their output logits with the softened prediction distributions generated by larger teacher ensembles.",
         "Compressing model ensembles into lightweight architectures utilizes distillation loss functions that match softened class probability distributions."),

        ("Positional Encoding Strategies",
         "Rotary position embeddings (RoPE) encode relative token distances directly into query and key representations via orthogonal rotation matrices, enhancing extrapolation to long sequence lengths.",
         "RoPE incorporates relative positional information into transformer self-attention by applying 2D rotation matrices to query and key vectors at each token index.",
         "Applying rotational transformation matrices to queries and keys in self-attention allows RoPE to inject relative position information with strong context extrapolation."),

        ("Active Learning Selection",
         "Uncertainty sampling strategies in active learning query human annotators for labels on unlabeled instances exhibiting maximal predictive entropy or minimal classification margins.",
         "Active learning prioritizes labeling the most informative unlabeled samples based on model uncertainty, such as highest prediction entropy or closest margin to decision boundaries.",
         "Selecting unannotated data points exhibiting highest prediction entropy minimizes labeling costs while accelerating active learning convergence."),

        ("Domain Adaptation via Adversarial Training",
         "Domain-adversarial neural networks learn domain-invariant representations by incorporating a gradient reversal layer that discourages the feature extractor from encoding domain-specific discriminative cues.",
         "DANN architectures achieve cross-domain transfer by training a feature extractor to deceive an auxiliary domain discriminator via gradient reversal.",
         "Cross-domain feature invariance is achieved using gradient reversal layers that penalize the representation learner for encoding domain-specific identifiers."),

        ("Neural Topic Modeling",
         "Variational autoencoding topic models approximate document-topic Dirichlet priors using continuous neural inference networks to discover coherent semantic themes across large text collections.",
         "Neural topic models utilize variational inference networks to parameterize document topic distributions over continuous latent spaces without Markov Chain Monte Carlo sampling.",
         "Discovering latent thematic topics in text corpora with variational autoencoders maps document word distributions into smooth continuous Dirichlet distributions."),

        ("Adversarial Robustness in Embeddings",
         "Projected gradient descent attacks perturb sentence embeddings along the direction of maximal loss gradient, identifying latent vulnerabilities in dense retrieval systems.",
         "PGD adversarial attacks iteratively compute gradient steps to construct embedding perturbations that maximize retrieval ranking loss.",
         "Targeted gradient ascent perturbations on sentence vector representations demonstrate significant ranking degradation in dense neural search engines."),

        ("Text Watermarking and Detection",
         "Statistical watermarking algorithms bias green-list token logits during autoregressive decoding to embed cryptographically verifiable detection signals into generated text.",
         "LLM watermarking alters token selection probabilities using a pseudo-random green list, enabling subsequent statistical verification of machine-generated text.",
         "Embedding verifiable detection fingerprints into language model output involves subtly boosting logits of pseudo-random green-list token subsets during decoding.")
    ]

    for idx, (topic, src, susp, adv) in enumerate(academic_topics, 1):
        pairs.append({
            "pair_id": f"ACAD_{idx:03d}",
            "domain": "academic",
            "benchmark": "PlagBench",
            "topic": topic,
            "source_text": src,
            "suspect_text": susp,
            "paraphrases": {
                "static_paraphrase": susp,
                "adversarial_paraphrase": adv,
                "claude_populated_paraphrase": ""
            },
            "metadata": {"domain_type": "Academic & Scientific NLP", "expected_challenge": "Dense embedding & lexical hybrid resistance"}
        })

    # -------------------------------------------------------------------------
    # DOMAIN 2: Obfuscated Sentences / PADBen (24 pairs)
    # -------------------------------------------------------------------------
    padben_topics = [
        ("Level 1: Synonym Substitution",
         "The financial regulator imposed strict penalties on the banking institution for systemic compliance failures.",
         "The monetary watchdog enforced severe sanctions on the banking firm for widespread regulatory non-compliance.",
         "Strict regulatory penalties were assessed against the financial corporation owing to comprehensive compliance defaults."),

        ("Level 2: Active-Passive Inversion",
         "The cybersecurity response team contained the data breach within four hours of initial detection.",
         "Within four hours of its discovery, the data breach was effectively contained by the cybersecurity response team.",
         "Swift containment of the system breach was accomplished by the security operations group within hours of initial compromise."),

        ("Level 3: Clause Restructuring & Repackaging",
         "Because the manufacturing plant suffered severe power outages, the company failed to meet its quarterly production quotas.",
         "The company missed its quarterly output targets as a direct result of critical electricity failures at its manufacturing facility.",
         "Quarterly manufacturing milestones were missed by the firm due to extensive electrical failures across production facilities."),

        ("Level 4: Nominalization & Syntactic Shift",
         "The research committee evaluated the clinical trial results thoroughly before recommending federal drug approval.",
         "A comprehensive evaluation of the clinical trial outcomes was performed by the research committee prior to endorsing government approval.",
         "Before issuing regulatory endorsement, the investigative panel conducted an exhaustive assessment of all clinical trial data."),

        ("Level 5: Multi-Hop Paraphrastic Laundering",
         "Autonomous driving algorithms require millions of diverse training miles to handle unpredictable edge-case scenarios safely.",
         "Safe handling of unpredictable corner cases necessitates that self-driving neural networks undergo millions of diverse real-world driving miles.",
         "To reliably negotiate rare road anomalies, self-driving vehicles must be trained across extensive mileage capturing varied edge-case distributions."),

        ("Level 3: Lexical Shuffling",
         "Renewable energy investments reached unprecedented global highs as solar and wind technology costs plummeted.",
         "Global capital allocation for clean energy attained historic peaks because expenditures for wind and solar equipment dropped drastically.",
         "Historic peaks in global green energy funding coincided with sharp declines in solar and wind capital expenditures."),

        ("Level 4: Discourse Marker Inversion",
         "Although the preliminary results were promising, the investigators warned that further longitudinal studies are required.",
         "The researchers cautioned that long-term investigations remain necessary, despite the encouraging nature of early findings.",
         "Notwithstanding favorable initial observations, the research team emphasized the imperative need for extended follow-up trials."),

        ("Level 5: Semantic Compression",
         "The modern telecommunications network utilizes fiber-optic infrastructure to transmit massive data volumes with minimal latency.",
         "High-throughput data transfer with ultra-low latency is achieved in contemporary telecommunication systems via fiber-optic cables.",
         "Contemporary telecom networks leverage high-bandwidth fiber-optic backbones to achieve rapid low-latency transmission."),

        ("Level 2: Phrase Splitting",
         "The board of directors approved the merger agreement unanimously, creating the largest logistics enterprise in South America.",
         "Unanimous approval of the merger contract was granted by the board, which established South America's foremost logistics conglomerate.",
         "With unanimous board consent on the merger pact, South America's largest freight and logistics entity was established."),

        ("Level 3: Antonym Inversion",
         "The patient exhibited rapid recovery without displaying any adverse reactions to the experimental pharmaceutical compound.",
         "No negative side effects were observed in the patient, who demonstrated a remarkably swift convalescence under the experimental therapy.",
         "Following experimental drug administration, the subject made a rapid recovery while remaining completely free of adverse symptoms."),

        ("Level 4: Metaphorical Recasting",
         "The sudden collapse of the regional bank triggered a liquidity crisis across the domestic financial market.",
         "Domestic financial markets experienced widespread liquidity shortages in the wake of the unexpected regional banking failure.",
         "A sudden failure of the regional lender sparked severe liquidity strain across broader financial markets."),

        ("Level 5: Information Reordering",
         "Researchers developed a novel biodegradable polymer that decomposes in seawater within ninety days without releasing microplastics.",
         "A new biodegradable plastic material that breaks down in ocean water in under three months without shedding microplastics was synthesized by scientists.",
         "Scientists formulated an innovative marine-degradable polymer that completely dissolves within three months avoiding microplastic contamination."),

        ("Level 1: Technical Synonymization",
         "The compiler optimized the abstract syntax tree to eliminate redundant computation and reduce executable size.",
         "Redundant execution was eliminated and binary footprint minimized when the compiler streamlined the abstract syntax representation.",
         "By refining the abstract syntax tree, the compiler pruned duplicate operations and compressed executable binaries."),

        ("Level 2: Passive Transformation",
         "The municipal council voted to allocate emergency relief funds to residents affected by the catastrophic flood.",
         "Emergency financial assistance was allocated by the city council to support citizens impacted by the disastrous flooding.",
         "Relief financing was granted by local municipal authorities to assist homeowners devastated by catastrophic inundation."),

        ("Level 3: Conjunction Shift",
         "Neither the software vendor nor the client took responsibility for the configuration defect that caused the outage.",
         "Both the software provider and the customer disclaimed liability for the configuration flaw leading to the service interruption.",
         "Responsibility for the configuration error causing system downtime was disclaimed by both the vendor and the enterprise client."),

        ("Level 4: Stylistic Shift",
         "The court affirmed that patent infringement occurs when an unauthorized product embodies every essential claim element.",
         "It was affirmed by the tribunal that unauthorized products incorporating all key claim limitations commit patent infringement.",
         "Judicial findings confirmed that incorporating each requisite claim element into an unlicensed device constitutes patent infringement."),

        ("Level 5: Contextual Relabeling",
         "The spacecraft executed a deceleration burn to insert itself into a stable elliptical orbit around Mars.",
         "A retrograde rocket firing was performed by the probe to achieve capture into an elliptical Martian orbital trajectory.",
         "By firing deceleration thrusters, the exploratory probe transitioned smoothly into a stable elliptical orbit around Mars."),

        ("Level 2: Structural Clause Reordering",
         "Unless international climate commitments are met, rising sea levels will displace millions of coastal inhabitants by mid-century.",
         "Millions of residents along coastlines face displacement by 2050 if global greenhouse reduction agreements are not fulfilled.",
         "Failure to satisfy multilateral emissions targets will result in the mass displacement of coastal populations by mid-century."),

        ("Level 3: Vocabulary Decoupling",
         "The biometric authentication system utilizes iris scanning to verify individual identity with high statistical certainty.",
         "High-confidence identity confirmation is performed by the biometric security platform through ocular pattern recognition.",
         "Iris verification technology enables the biometric access system to authenticate individual identities with extreme statistical precision."),

        ("Level 4: Inverted Syntax",
         "Rarely has a volcanic eruption produced such widespread atmospheric disruption in such a brief timeframe.",
         "Seldom has atmospheric equilibrium been disturbed so broadly and rapidly by a single volcanic event.",
         "A volcanic eruption has rarely caused such extensive and immediate climate and atmospheric disturbances."),

        ("Level 5: Deep Paraphrastic Scrubbing",
         "The distributed database ensures ACID compliance across partitioned clusters using multi-version concurrency control.",
         "MVCC mechanisms are employed by the partitioned database architecture to maintain strict ACID transaction guarantees.",
         "To preserve ACID transaction integrity across sharded nodes, the distributed database relies on multi-version concurrency protocols."),

        ("Level 1: Lexical Laundering",
         "The educational institution implemented mandatory academic integrity tutorials to reduce student plagiarism incidents.",
         "Compulsory honesty workshops were introduced by the university to curb instances of student cheating and uncredited copying.",
         "To diminish uncredited text duplication, the academic department instituted mandatory training modules on scholarly integrity."),

        ("Level 3: Cause-Effect Swapping",
         "Deforestation in the Amazon basin accelerated due to expanding commercial agricultural operations and illegal logging.",
         "Expanding commercial farming and unauthorized timber harvesting drove an increase in forest loss across the Amazon.",
         "Rising commercial farming and illicit logging activities served as the primary drivers of accelerated Amazonian deforestation."),

        ("Level 4: Complex Synthesis",
         "The antitrust regulator blocked the telecommunications acquisition, concluding that diminished competition would harm consumers.",
         "The telecom buyout was rejected by antitrust authorities on the grounds that reduced market competition would negatively affect consumers.",
         "Antitrust regulators prohibited the telecommunications takeover after determining that diminished market competition would disadvantage consumer welfare.")
    ]

    for idx, (topic, src, susp, adv) in enumerate(padben_topics, 1):
        pairs.append({
            "pair_id": f"PAD_{idx:03d}",
            "domain": "padben",
            "benchmark": "PADBen",
            "topic": topic,
            "source_text": src,
            "suspect_text": susp,
            "paraphrases": {
                "static_paraphrase": susp,
                "adversarial_paraphrase": adv,
                "claude_populated_paraphrase": ""
            },
            "metadata": {"domain_type": "Adversarial Paraphrasing", "obfuscation_type": topic}
        })

    # -------------------------------------------------------------------------
    # DOMAIN 3: Legal Court Corpus (24 pairs)
    # -------------------------------------------------------------------------
    legal_topics = [
        ("Labor Law & Occupational Liability",
         "The defendant employer failed to provide adequate occupational safety equipment mandated by Article 157 of the Labor Code, thereby directly contributing to the workplace accident sustained by the plaintiff employee. Consequently, the regional labor tribunal holds the employer strictly liable for compensatory damages, medical reimbursement, and moral pain suffered as established under statutory civil provisions.",
         "The respondent corporation neglected its statutory obligation to supply required occupational safety gear under Article 157 of the Labor Regulations, which was the proximate cause of the petitioner's industrial injury. Therefore, the labor court establishes strict employer liability for pecuniary restitution, accrued medical expenses, and non-pecuniary damages pursuant to established civil jurisprudence.",
         "Owing to the respondent's breach of Article 157 workplace safety mandates, liability for the claimant's industrial injury is affirmed. The regional tribunal awards compensatory restitution, healthcare expense reimbursement, and moral damages under established civil law principles."),

        ("Commercial Non-Disclosure Covenants",
         "Under the non-disclosure covenant set forth in Section 4.2, the receiving party agrees to hold all proprietary technical data and trade secrets in strict confidence. Any unauthorized disclosure, reverse engineering, or third-party dissemination shall constitute an immediate material breach entitling the disclosing party to injunctive relief and statutory liquidated damages.",
         "Pursuant to the confidentiality provisions in Section 4.2, the recipient party undertakes to maintain all proprietary technical specifications and commercial secrets under rigorous secrecy. Unauthorized disclosure, decompilation, or dissemination to unauthorized entities represents a material contractual default, granting the disclosing entity the right to preliminary injunctive remedies and specified damages.",
         "Clause 4.2 obligates the recipient to preserve strict secrecy regarding technical trade secrets. Any unlicensed disclosure or reverse engineering establishes an actionable material default, affording the disclosing corporation equitable injunctions and predetermined liquidated damages."),

        ("Intellectual Property & Patent Infringement",
         "In patent infringement litigation, the doctrine of equivalents permits a finding of infringement when an accused device performs substantially the same function in substantially the same way to achieve substantially the same result as the claimed invention.",
         "Under patent jurisprudence, the doctrine of equivalents establishes infringement if the contested product executes substantially identical functions through substantially equivalent mechanisms to obtain substantially the same technical outcome.",
         "Equivalence doctrine allows patent holders to demonstrate infringement where an unauthorized device achieves substantially identical outcomes through substantially identical operational modalities as described in the patent claims."),

        ("Breach of Commercial Fiduciary Duty",
         "Corporate directors owe an unwavering duty of loyalty and care to the enterprise, requiring them to subordinate private pecuniary interests to the collective welfare of shareholders during corporate acquisitions and reorganizations.",
         "Board members are bound by fiduciary obligations of care and loyalty, obliging them to prioritize shareholder interests over personal financial gain when negotiating corporate restructuring and asset acquisitions.",
         "Fiduciary duties dictate that corporate officers must prioritize collective shareholder interests above personal enrichment when conducting mergers, asset transfers, and corporate restructurings."),

        ("Statutory Product Liability",
         "Under strict product liability principles, the manufacturer is liable for physical harm caused by a design defect, irrespective of whether the manufacturer exercised all reasonable care in the fabrication and marketing of the product.",
         "Strict liability doctrines hold product manufacturers responsible for consumer injuries resulting from design flaws, regardless of whether reasonable diligence was exercised during manufacturing and commercial distribution.",
         "Product manufacturers bear strict civil liability for injuries caused by inherent design flaws, notwithstanding adherence to customary manufacturing care and regulatory standards."),

        ("International Commercial Arbitration",
         "The arbitration clause stipulates that all disputes arising out of or in connection with the present contract shall be finally settled under the Rules of Arbitration of the International Chamber of Commerce by three arbitrators.",
         "Contractual disputes stemming from this agreement shall be definitively resolved pursuant to the ICC Arbitration Rules before a tribunal comprising three appointed arbitrators.",
         "All controversies connected to this contract shall be submitted to binding ICC arbitration before a three-member arbitral panel in accordance with institutional rules."),

        ("Environmental Statutory Liability",
         "Industrial polluters bear joint and several strict liability for hazardous waste remediation costs and natural resource damages under the Comprehensive Environmental Response, Compensation, and Liability Act.",
         "Under CERCLA regulations, industrial entities responsible for hazardous contamination are held jointly and strictly liable for environmental cleanup expenses and ecological restoration.",
         "CERCLA establishes joint and strict liability for industrial parties regarding toxic substance containment, contaminated site remediation, and environmental damages."),

        ("Antitrust & Monopolistic Practices",
         "The statutory prohibition against tying arrangements prohibits market participants with substantial market power from conditioning the sale of a tying product on the customer's purchase of a distinct tied product.",
         "Antitrust law prevents dominant market entities from requiring consumers to purchase a secondary tied product as a prerequisite for acquiring the primary tying good.",
         "Conditioning the availability of a primary commercial good on the mandatory purchase of an ancillary product constitutes an unlawful tying arrangement under competition law."),

        ("Force Majeure Contractual Clauses",
         "Neither party shall be held liable for failure to perform contractual obligations if such non-performance results from events beyond reasonable control, including acts of God, armed conflict, civil unrest, or governmental embargoes.",
         "Performance obligations are excused where default is attributable to uncontrollable force majeure occurrences, including natural disasters, military hostilities, civil insurrections, or state sanctions.",
         "Inability to satisfy contractual commitments is excused under force majeure principles when caused by uncontrollable events such as armed conflict, severe natural disasters, or statutory prohibitions."),

        ("Indemnification & Hold Harmless",
         "The contractor agrees to defend, indemnify, and hold harmless the property owner from any claims, damages, liabilities, and legal fees arising out of the contractor's negligent acts or omissions on the job site.",
         "The property owner shall be indemnified, defended, and held harmless by the contractor against all legal claims, losses, and attorney expenses resulting from the contractor's onsite negligence.",
         "The service provider covenants to indemnify and defend the owner against all third-party liabilities and litigation costs occasioned by negligent acts during project execution."),

        ("Employment Non-Compete Enforceability",
         "Covenants not to compete are enforceable only to the extent they protect legitimate business interests, are reasonable in geographic scope and temporal duration, and do not impose undue hardship on the former employee.",
         "Post-employment non-compete agreements are legally binding solely if reasonably restricted in geographic territory and duration while safeguarding valid commercial interests without unduly burdening the worker.",
         "Enforceability of employee non-compete covenants requires reasonable geographical and temporal limits tailored strictly to protect legitimate trade secrets without unduly restraining employment."),

        ("Securities Fraud & Material Misstatement",
         "To establish liability under Rule 10b-5, the plaintiff must prove that the defendant made a material misrepresentation with scienter in connection with the purchase or sale of a security, resulting in economic loss.",
         "Securities fraud claims under Section 10(b) require demonstrating that a material misstatement was issued with fraudulent intent, directly causing investor financial loss in security transactions.",
         "Prevailing on securities fraud claims necessitates establishing that material false statements were made intentionally or recklessly in connection with security trading, causing investor losses."),

        ("Healthcare Informed Consent",
         "The doctrine of informed consent requires medical practitioners to disclose all material risks, potential complications, and therapeutic alternatives before obtaining patient authorization for invasive surgical interventions.",
         "Physicians are legally required under informed consent standards to communicate significant risks and alternative treatments to patients prior to executing invasive procedures.",
         "Valid consent for invasive medical procedures requires practitioners to inform patients of significant risks, therapeutic options, and potential complications prior to treatment."),

        ("Consumer Protection & Unfair Practices",
         "Deceptive commercial practices that mislead reasonable consumers regarding product efficacy, pricing, or origin constitute statutory violations enforceable by the Federal Trade Commission.",
         "Misleading representations concerning product performance, cost, or provenance that deceive typical consumers violate statutory consumer protection mandates enforced by the FTC.",
         "Commercial deception regarding merchandise characteristics, pricing, or origins is prohibited under federal consumer protection laws and subject to FTC regulatory action."),

        ("Data Privacy & GDPR Compliance",
         "Data controllers must implement appropriate technical and organizational measures to ensure that personal data processing complies with principles of data minimization, purpose limitation, and storage limitation.",
         "GDPR mandates that data processors adopt robust technical safeguards ensuring personal information handling adheres to strict purpose boundaries, minimal collection, and retention limits.",
         "Compliance with statutory data privacy principles obligates organizations to maintain organizational controls enforcing data minimization, specific purpose limitation, and defined retention periods."),

        ("Mortgage Foreclosure Procedure",
         "Upon borrower default under the promissory note, the mortgagee must issue formal notice of default providing thirty days to cure the delinquency prior to commencing judicial foreclosure proceedings.",
         "Following mortgage payment default, the lender is required to serve a formal notice granting a thirty-day cure period before initiating court foreclosure actions.",
         "Statutory foreclosure rules require financial lenders to deliver formal delinquency notices allowing thirty days for remediation before initiating property foreclosure proceedings."),

        ("Copyright Fair Use Doctrine",
         "Evaluating statutory fair use requires courts to weigh the purpose of the use, the nature of the copyrighted work, the amount taken, and the market effect on the original creative expression.",
         "Fair use determination involves judicial consideration of four statutory factors: transformative purpose, copyrighted work characteristics, proportion utilized, and commercial market impact.",
         "Assessing fair use defenses involves examining the transformative purpose, work nature, quantity borrowed, and potential economic harm to the market for the original copyrighted material."),

        ("Commercial Lease Termination",
         "The landlord may terminate the commercial tenancy if the tenant fails to pay base rent within five business days following written notice of monetary default.",
         "Commercial lease agreements may be cancelled by the lessor if overdue rental sums remain unpaid after five business days following formal default notification.",
         "Lessor rights to cancel commercial property leases vest if the tenant neglects to remedy rental defaults within five business days of written notice."),

        ("Statute of Limitations & Tolling",
         "The statutory limitation period for breach of written contract claims is four years from the date of the breach, subject to equitable tolling where the injury was fraudulently concealed.",
         "Actions for written contract default must be filed within four years of the breach, with limitation clocks tolled during periods of fraudulent concealment.",
         "Claims for breach of written contracts expire four years after default, unless the limitation statutory window is equitably tolled due to fraudulent concealment."),

        ("Corporate Dissolution & Liquidation",
         "Upon judicial dissolution of a corporation, company assets must be applied first toward satisfying creditor obligations before any residual distributions are allocated to equity shareholders.",
         "During corporate liquidation proceedings, available assets must be directed to extinguish creditor debts prior to distributing remaining funds among equity investors.",
         "Corporate dissolution statutes prescribe that creditor claims hold absolute priority over equity shareholder distributions during asset liquidation proceedings."),

        ("Construction Defect Warranty",
         "The general contractor warrants that all structural work performed under the construction agreement shall be free from latent defects and executed in accordance with applicable building codes for ten years.",
         "Construction contracts impose a ten-year warranty on general contractors guaranteeing structural integrity and full compliance with municipal building codes.",
         "Structural work executed under the construction agreement carries a ten-year warranty against latent defects, binding the contractor to municipal code standards."),

        ("Trademark Likelihood of Confusion",
         "Establishing trademark infringement requires demonstrating that the concurrent commercial use of similar marks is likely to cause consumer confusion as to product source or affiliation.",
         "Trademark infringement claims succeed upon proving that comparable brand identifiers create a likelihood of consumer confusion regarding the commercial origin of goods.",
         "Liability for trademark infringement arises when similarity between commercial marks creates consumer confusion concerning product source or corporate sponsorship."),

        ("Arbitration Award Confirmation",
         "Judicial review of a final arbitral award is strictly limited, and courts must confirm the award unless statutory grounds such as arbitrator corruption or manifest disregard of law are proven.",
         "Courts possess narrow authority to vacate arbitral decisions, with mandatory confirmation required unless evidence demonstrates arbitrator fraud or intentional disregard of governing law.",
         "Judicial confirmation of arbitral awards is mandatory under federal arbitration statutes absent clear proof of arbitrator misconduct, fraud, or manifest disregard of law."),

        ("Whistleblower Retaliation Protection",
         "The Sarbanes-Oxley Act prohibits employers from discharging, demoting, or harassing employees who provide information regarding corporate financial fraud to federal regulatory agencies.",
         "Federal law protects corporate whistleblowers from retaliatory termination, demotion, or workplace hostility for reporting accounting irregularities to regulatory authorities.",
         "Statutory whistleblower provisions shield employees from retaliatory discharge or workplace discrimination when disclosing financial irregularities to government regulators.")
    ]

    for idx, (topic, src, susp, adv) in enumerate(legal_topics, 1):
        pairs.append({
            "pair_id": f"LEGAL_{idx:03d}",
            "domain": "legal",
            "benchmark": "Legal Court Corpus",
            "topic": topic,
            "source_text": src,
            "suspect_text": susp,
            "paraphrases": {
                "static_paraphrase": susp,
                "adversarial_paraphrase": adv,
                "claude_populated_paraphrase": ""
            },
            "metadata": {"domain_type": "Legal Jurisprudence & Contracts", "legal_area": topic}
        })

    # -------------------------------------------------------------------------
    # DOMAIN 4: Scientific / SciDocs & CSFCUBE (24 pairs)
    # -------------------------------------------------------------------------
    scidocs_topics = [
        ("CRISPR Genomic Cleavage",
         "CRISPR-Cas9 endonuclease complexes achieve targeted genomic editing by recognizing specific protospacer adjacent motifs (PAM) and inducing double-strand DNA breaks. Cellular repair pathways, specifically non-homologous end joining (NHEJ) and homology-directed repair (HDR), subsequently facilitate precise gene knockouts or targeted nucleotide insertions.",
         "The CRISPR-Cas9 ribonucleoprotein system executes site-specific genetic modification by identifying protospacer adjacent motif (PAM) sequences and introducing double-stranded DNA cleavage. Endogenous repair mechanisms, namely non-homologous end joining (NHEJ) and homology-directed repair (HDR), mediate subsequent targeted gene disruption or precise sequence knock-ins.",
         "Targeted genomic alterations are mediated by Cas9 ribonucleoproteins upon PAM recognition and subsequent double-strand DNA cleavage. Host repair machinery, involving NHEJ and HDR pathways, executes targeted nucleotide insertions or gene disruptions."),

        ("Topological Quantum Gates",
         "Topological quantum computation exploits non-Abelian anyons in two-dimensional electron gases to perform fault-tolerant quantum gate operations. By braiding world-lines of quasi-particles in spacetime, quantum information is encoded non-locally, conferring intrinsic immunity against local environmental decoherence.",
         "Topological quantum computers utilize non-Abelian anyonic excitations in 2D electron systems to execute error-resilient quantum logic gates. Spatiotemporal braiding of quasi-particle trajectories encodes quantum states in a non-local topological topology, providing inherent protection against environmental decoherence.",
         "Non-Abelian anyons in 2D electron gases facilitate fault-tolerant quantum operations through quasi-particle braiding. Non-local state encoding provides topological immunity against localized environmental decoherence."),

        ("Superconducting Metamaterials",
         "High-temperature cuprate superconductors exhibit anomalous normal-state resistivity that scales linearly with temperature down to the superconducting transition, challenging conventional Fermi liquid transport theories.",
         "Cuprate high-Tc superconductors display strange metal transport properties where electrical resistivity varies linearly with temperature until reaching the critical transition, violating standard Landau Fermi-liquid predictions.",
         "Linear temperature dependence of electrical resistivity in normal-state cuprates deviates from Landau Fermi-liquid predictions, indicating non-quasiparticle transport mechanisms above the critical transition."),

        ("Perovskite Solar Cells",
         "Organometal halide perovskites demonstrate exceptional photovoltaic conversion efficiencies owing to their high absorption coefficients, long carrier diffusion lengths, and low non-radiative recombination rates.",
         "Halide perovskite semiconductor materials achieve superior solar cell performance due to strong optical absorption, extended carrier lifespans, and minimal non-radiative recombination losses.",
         "Exceptional photovoltaic performance in halide perovskite solar absorbers stems from high optical absorption coefficients combined with extended carrier diffusion lengths and reduced recombination."),

        ("Graphene Nanoribbon Transport",
         "Quantum confinement in armchair graphene nanoribbons opens an electronic bandgap that is inversely proportional to the ribbon width, enabling room-temperature field-effect transistor operation.",
         "Armchair-edge graphene nanoribbons generate a finite energy bandgap via quantum confinement effects inversely scaled with ribbon width, facilitating field-effect transistor applications at ambient temperatures.",
         "Engineering finite energy bandgaps in armchair graphene nanoribbons via width-dependent quantum confinement enables room-temperature field-effect transistor architectures."),

        ("Metabolic Flux in Cancer",
         "Oncogenic KRAS mutations rewire cellular metabolism by upregulating glutaminolysis and glycolysis to sustain rapid biosynthesis and redox balance in pancreatic ductal adenocarcinoma cells.",
         "Pancreatic cancer cells harboring oncogenic KRAS reprogram metabolic pathways, enhancing glucose uptake and glutamine consumption to support macromolecular synthesis and antioxidant homeostasis.",
         "Reprogrammed nutrient flux driven by oncogenic KRAS accelerates glycolysis and glutamine metabolism to maintain redox balance and nucleotide biosynthesis in malignant pancreatic cells."),

        ("Gravitational Wave Interferometry",
         "Laser interferometer observatories detect gravitational waves from binary black hole coalescences by measuring differential arm length perturbations using frequency-stabilized lasers and quantum squeezed light.",
         "Gravitational wave detectors capture ripples in spacetime from merging black holes by monitoring picometer arm length variations with stabilized laser interferometers and squeezed vacuum states.",
         "Detecting metric spacetime perturbations from compact binary coalescences relies on optical interferometers employing frequency-stabilized lasers and squeezed vacuum states to measure arm length variations."),

        ("Solid-State Electrolytes",
         "Lithium superionic conductor solid electrolytes provide high ionic conductivity exceeding ten millisiemens per centimeter, preventing dendrite penetration while enabling safe high-energy lithium metal batteries.",
         "Solid-state lithium conductors deliver ionic transport rates above 10 mS/cm, suppressing lithium dendrite growth and facilitating stable high-voltage metal battery operation.",
         "Solid superionic lithium conductors achieving high ionic transport mitigate dendrite formation, enabling safe integration of high-energy lithium metal anodes."),

        ("Optogenetic Neural Modulation",
         "Channelrhodopsin-2 expression in specific cortical neuronal subpopulations allows millisecond-precision optical excitation of target circuits upon blue light illumination in mammalian models.",
         "Targeted expression of light-gated Channelrhodopsin-2 proteins enables precise temporal activation of specific neural circuits via blue wavelength photo-stimulation.",
         "Expressing light-sensitive opsins in distinct neuronal sub-types allows millisecond-scale optical activation of targeted mammalian neural circuits with blue light pulses."),

        ("Catalytic Nitrogen Reduction",
         "Electrochemical ambient nitrogen reduction utilizes single-atom ruthenium catalysts anchored on nitrogen-doped carbon matrices to break the nitrogen-nitrogen triple bond at low overpotentials.",
         "Ambient electrocatalytic nitrogen fixation leverages atomically dispersed ruthenium on carbon substrates to facilitate N2 cleavage under modest overpotential conditions.",
         "Single-atom ruthenium catalysts supported on nitrogen-doped carbon scaffolds facilitate ambient electrochemical nitrogen reduction by lowering the activation barrier for N2 triple-bond cleavage."),

        ("Stochastic Resonance in Nanodevices",
         "Stochastic resonance phenomena in bistable magnetic tunnel junctions enhance sub-threshold signal detection through constructive coupling between ambient thermal noise and weak external periodic inputs.",
         "Bistable magnetic tunnel junctions exploit stochastic resonance, where optimal thermal noise levels amplify weak periodic signals exceeding the switching threshold.",
         "Constructive interaction between thermal noise and sub-threshold periodic signals in bistable spintronic junctions enhances weak signal detection via stochastic resonance."),

        ("Synthetic Microfluidic Biology",
         "Droplet microfluidics encapsulates single bacterial cells into picoliter aqueous droplets, allowing ultra-high-throughput enzymatic screening and directed evolution at kilohertz frequencies.",
         "Picoliter droplet microfluidic systems isolate individual microbes in emulsion droplets to conduct ultra-fast enzyme screening assays and evolutionary selections.",
         "Encapsulating single microbial cells in picoliter emulsion droplets enables high-throughput enzymatic profiling and directed evolution at kilohertz assay rates."),

        ("Dark Matter Particle Signatures",
         "Liquid xenon time projection chambers detect weakly interacting massive particles (WIMPs) through simultaneous measurement of prompt scintillation photons and delayed ionization electrons.",
         "WIMP dark matter searches utilize dual-phase xenon detectors to identify particle recoil events via paired primary scintillation and secondary ionization charge signals.",
         "Dual-phase xenon time projection detectors identify potential WIMP dark matter scattering through concurrent observation of primary scintillation and drifted ionization electrons."),

        ("Ribosome Profiling in Translation",
         "Ribo-seq quantifies global translation dynamics at single-nucleotide resolution by deep sequencing messenger RNA fragments protected from ribonuclease digestion by actively translating ribosomes.",
         "Ribosome profiling measures in vivo protein synthesis rates genome-wide by isolating and sequencing nuclease-protected mRNA footprints during translation.",
         "Genome-wide translational kinetics are quantified at single-nucleotide resolution by isolating and sequencing ribonuclease-protected mRNA footprints from translating ribosomes."),

        ("Spintronic Spin-Orbit Torque",
         "Spin-orbit torques generated by the spin Hall effect in heavy metal and ferromagnet heterostructures enable ultra-fast, energy-efficient magnetization switching for non-volatile magnetic memory.",
         "Heavy metal/ferromagnet bilayers utilize spin Hall effects to generate spin-orbit torques that drive fast magnetic reversal in non-volatile spintronic storage devices.",
         "Spin Hall currents in heavy-metal ferromagnet bilayers generate spin-orbit torques, driving efficient sub-nanosecond magnetization switching for non-volatile MRAM."),

        ("Synthetic Organic Photocatalysis",
         "Dual photoredox and nickel catalytic systems couple aryl halides with alkyl carboxylic acids via decarboxylative radical generation under visible light irradiation.",
         "Metallaphotoredox catalysis merges visible-light ruthenium/iridium dyes with nickel catalysts to achieve cross-coupling between aryl halides and carboxylic acid derivatives.",
         "Combining visible-light photoredox catalysts with nickel complexes enables cross-coupling of aryl halides through light-driven decarboxylative radical generation."),

        ("Exoplanet Atmospheric Spectroscopy",
         "Transmission spectroscopy during exoplanet transits measures wavelength-dependent stellar absorption to identify atmospheric chemical species such as water vapor, methane, and carbon dioxide.",
         "Transit transmission spectra reveal exoplanetary atmospheric composition by detecting chemical absorption signatures of H2O, CH4, and CO2 in filtered starlight.",
         "Filtering starlight through exoplanetary atmospheres during transit events reveals chemical constituents including water vapor, carbon dioxide, and methane via transmission spectroscopy."),

        ("Thermodynamic Nanoscale Heat Transport",
         "Phonon boundary scattering in silicon phononic crystal membranes reduces thermal conductivity below the alloy limit while maintaining electrical conductivity for thermoelectric energy harvesting.",
         "Phononic crystal nanostructures suppress thermal conduction in silicon via boundary scattering without degrading electrical charge transport for thermoelectric conversion.",
         "Engineering phononic crystal membranes induces coherent phonon boundary scattering, depressing thermal conductivity below bulk limits while preserving electrical transport."),

        ("Macromolecular Cryo-EM Reconstruction",
         "Single-particle cryogenic electron microscopy reconstructs near-atomic resolution 3D structural models of macromolecular complexes from thousands of individual 2D particle projections in vitreous ice.",
         "Cryo-EM determines near-atomic macromolecular structures by computational alignment and 3D reconstruction of multiple 2D single-particle micrographs captured in amorphous ice.",
         "Averaging thousands of 2D cryogenic electron micrographs of vitrified macromolecular specimens enables 3D near-atomic structural reconstructions of complex proteins."),

        ("Non-Equilibrium Plasma Chemistry",
         "Atmospheric pressure non-thermal plasmas generate reactive oxygen and nitrogen species at ambient temperatures, enabling targeted decontamination of heat-sensitive biomedical surfaces.",
         "Non-equilibrium cold atmospheric plasmas produce reactive nitrogen and oxygen radicals to sterilize sensitive medical materials without thermal damage.",
         "Generating reactive oxygen and nitrogen species in cold atmospheric plasmas enables rapid sterilization of heat-sensitive biomedical materials at room temperature."),

        ("Single-Molecule FRET Kinetics",
         "Single-molecule fluorescence resonance energy transfer (smFRET) tracks conformational transitions of protein complexes in real time by measuring distance-dependent non-radiative energy transfer.",
         "smFRET spectroscopy monitors real-time macromolecular structural dynamics by detecting distance-dependent dipole energy transfer between donor and acceptor fluorophores.",
         "Real-time conformational fluctuations of single biomolecules are resolved by measuring distance-sensitive non-radiative dipole coupling between fluorescent probes in smFRET."),

        ("Memristive Neuromorphic Synapses",
         "Filamentary memristive crossbar arrays emulate biological synaptic plasticity through continuous analog conductance modulation driven by electric field-induced oxygen vacancy migration.",
         "Oxide-based memristor crossbars replicate synaptic weight updates via field-driven migration of oxygen vacancies that modulate device electrical conductance.",
         "Analog conductance modulation in metal-oxide memristor arrays emulates biological synaptic plasticity via voltage-driven redistribution of oxygen vacancy filaments."),

        ("Atmospheric Aerosol Nucleation",
         "Sulfuric acid and organic vapor clusters undergo multi-component molecular nucleation in the planetary boundary layer, serving as critical cloud condensation nuclei governing climate forcing.",
         "New particle formation in the atmosphere proceeds through co-nucleation of sulfuric acid and biogenic organics, producing aerosol particles that act as cloud condensation nuclei.",
         "Molecular clustering of sulfuric acid and oxidized organic vapors drives atmospheric aerosol nucleation, producing cloud condensation nuclei that modulate radiative forcing."),

        ("Synthetic Gene Regulatory Circuits",
         "Synthetic genetic toggle switches constructed from mutually repressing transcriptional repressors establish bistable steady states in engineered Escherichia coli host organisms.",
         "Bistable gene regulatory networks built with reciprocal repressor proteins create robust genetic memory switches in recombinant E. coli strains.",
         "Mutually inhibitory transcriptional repressors integrated into synthetic genetic circuits generate bistable regulatory states with robust epigenetic memory in microbial hosts.")
    ]

    for idx, (topic, src, susp, adv) in enumerate(scidocs_topics, 1):
        pairs.append({
            "pair_id": f"SCI_{idx:03d}",
            "domain": "scidocs",
            "benchmark": "SciDocs",
            "topic": topic,
            "source_text": src,
            "suspect_text": susp,
            "paraphrases": {
                "static_paraphrase": susp,
                "adversarial_paraphrase": adv,
                "claude_populated_paraphrase": ""
            },
            "metadata": {"domain_type": "Scientific & Technical Literature", "scientific_field": topic}
        })

    # -------------------------------------------------------------------------
    # DOMAIN 5: Journalistic / News & SemEval-2022 Task 8 (24 pairs)
    # -------------------------------------------------------------------------
    news_topics = [
        ("Central Bank Interest Rates",
         "The central monetary authority unexpectedly raised baseline interest rates by fifty basis points on Thursday, citing persistent inflationary pressures and volatile energy commodity prices across international markets. Equity indices plummeted following the press briefing as investors weighed the likelihood of an impending recession.",
         "In an unpredicted policy shift on Thursday, the central banking authority increased the benchmark interest rate by 50 basis points due to stubborn inflation and turbulent global energy markets. Financial markets dropped sharply during the subsequent press conference as traders assessed escalating recession risks.",
         "In response to persistent inflation and energy price volatility, the central bank implemented a surprise 50 basis point rate hike on Thursday. Equity markets experienced steep declines as market participants factored in increased probabilities of an economic downturn."),

        ("Next-Gen Meteorological Satellite",
         "Space exploration officials announced the successful launch of the next-generation meteorological satellite from the coastal spaceport early Tuesday morning. The orbital platform is equipped with advanced synthetic aperture radar to deliver real-time atmospheric moisture mapping and hurricane tracking data.",
         "Aerospace officials confirmed the flawless orbital launch of an advanced weather monitoring satellite from the maritime launch facility on Tuesday dawn. Featuring cutting-edge synthetic aperture radar sensors, the spacecraft will provide continuous atmospheric moisture observations and tropical storm tracking.",
         "A next-generation weather observation satellite was successfully inserted into orbit from the coastal launch complex early Tuesday. Outfitted with synthetic aperture radar instruments, the spacecraft provides continuous atmospheric moisture telemetry and cyclone monitoring."),

        ("Semiconductor Manufacturing Subsidies",
         "The trade ministry approved a multi-billion dollar subsidy package to construct domestic semiconductor fabrication plants, aiming to insulate critical automotive and defense industries from global supply chain disruptions.",
         "Government officials authorized billions in financial incentives for local semiconductor manufacturing facilities to protect automotive and defense sectors against foreign supply chain shocks.",
         "To safeguard defense and automotive supply chains against external shocks, the commerce ministry allocated billions in public capital to fund domestic semiconductor foundries."),

        ("Electric Vehicle Battery Mandates",
         "Municipal regulators announced ambitious environmental guidelines requiring all public transit buses and commercial taxi fleets to transition to zero-emission battery electric drivetrains by the end of the decade.",
         "City authorities enacted regulations mandating that commercial taxis and municipal bus networks adopt zero-emission electric vehicles before 2030.",
         "Local transit authorities instituted statutory environmental standards requiring commercial fleets and city buses to achieve full electrification by 2030."),

        ("Diplomatic Peace Negotiations",
         "Diplomatic envoys concluded three days of bilateral negotiations in Geneva, reaching a preliminary framework agreement regarding maritime boundary demarcation and disputed offshore resource extraction rights.",
         "Envoys finalized a tentative agreement in Geneva governing maritime borders and offshore drilling entitlements following three days of intensive bilateral talks.",
         "Following three days of bilateral discussions in Geneva, diplomatic representatives finalized a provisional treaty defining maritime boundaries and offshore natural resource rights."),

        ("Cybersecurity Critical Infrastructure Threat",
         "Intelligence agencies issued a joint advisory warning of state-sponsored cyber intrusions targeting power grid distribution nodes and water treatment facilities using polymorphic credential-harvesting malware.",
         "Security authorities alerted critical infrastructure operators to nation-state cyber attacks attempting to infiltrate electrical grids and municipal water utilities with evasive malware.",
         "A joint security bulletin warned that foreign state-backed actors are targeting energy grids and municipal water treatment plants with advanced credential-harvesting malware."),

        ("Commercial Aviation Strike Action",
         "Airline pilot unions declared an indefinite nationwide strike starting midnight, forcing major carriers to cancel hundreds of international flights and stranding thousands of holiday passengers.",
         "A nationwide walkout called by pilot associations forced major airlines to ground hundreds of overseas flights, creating widespread travel disruption for thousands of passengers.",
         "Pilot union strike action commenced at midnight, prompting airlines to cancel hundreds of long-haul routes and stranding thousands of travelers nationwide."),

        ("Pharmaceutical Vaccine Approval",
         "Federal healthcare regulators granted expedited approval for the novel mRNA vaccine targeting emerging seasonal influenza variants, citing robust efficacy in Phase III randomized clinical trials.",
         "Health authorities authorized emergency rollout of a new mRNA flu vaccine following strong protection rates demonstrated across large-scale Phase III human trials.",
         "Accelerated regulatory clearance was awarded to an updated mRNA influenza vaccine based on favorable clinical trial data demonstrating high protective efficacy."),

        ("Deep-Sea Mining Environmental Debate",
         "Oceanographic researchers presented evidence to the maritime convention demonstrating that deep-sea mineral extraction causes irreversible disruption to benthic ecosystems and hydrothermal vent communities.",
         "Scientists informed the international maritime assembly that seafloor mining operations inflict lasting ecological damage on deep-sea habitats and hydrothermal vent fauna.",
         "Marine scientists warned the international ocean authority that seabed mineral harvesting causes permanent ecological degradation to benthic biodiversity and hydrothermal ecosystems."),

        ("Retail E-Commerce Antitrust Probe",
         "Competition authorities launched a formal investigation into the dominant e-commerce platform over allegations of favoring its private-label brands in algorithmic product search rankings.",
         "Antitrust regulators initiated a probe examining whether the leading online marketplace unfairly boosts its own branded products in algorithmic customer search results.",
         "Antitrust regulators opened formal proceedings against the e-commerce conglomerate over alleged search ranking discrimination favoring in-house private-label merchandise."),

        ("Renewable Offshore Wind Farm",
         "Construction commenced on the largest offshore wind farm in the North Sea, projected to generate five gigawatts of clean electricity and power four million residential households upon completion.",
         "Engineering teams began installation of a massive North Sea offshore wind project designed to produce 5 GW of renewable energy for four million homes.",
         "Work began on a 5-gigawatt offshore wind development in the North Sea, designed to supply zero-carbon electrical power to four million homes upon commissioning."),

        ("Corporate Accounting Scandal",
         "Financial regulators suspended trading of the multinational conglomerate's shares after forensic audits uncovered billions of dollars in off-balance-sheet liabilities and fabricated revenues.",
         "Securities regulators halted trading of the multinational firm following an audit that exposed fictitious revenue reporting and billions in concealed debt obligations.",
         "Trading in the multinational conglomerate was suspended following forensic disclosures of multi-billion dollar concealed liabilities and fabricated accounting revenue."),

        ("Global Food Commodity Inflation",
         "Agricultural economists warned that prolonged drought across major grain-producing regions will drive global wheat and corn prices to record highs, exacerbating food security challenges in developing economies.",
         "Severe regional droughts are projected to propel international corn and wheat prices to record levels, worsening nutritional vulnerabilities across developing nations.",
         "Persistent drought in major agricultural basins is expected to push global grain prices to historic highs, intensifying food insecurity across vulnerable developing regions."),

        ("Artificial Intelligence Copyright Lawsuit",
         "A coalition of prominent authors filed a class-action lawsuit against the artificial intelligence enterprise, alleging unauthorized scraping of copyrighted literary works for training neural language models.",
         "Leading authors initiated class-action litigation against an AI firm for unpermitted use of copyrighted books and literary texts to train generative language algorithms.",
         "A major author association initiated class-action litigation alleging that the generative AI company scraped copyrighted literary volumes without authorization to train foundation models."),

        ("Space Telescope Deep-Field Discovery",
         "Astronomers operating the orbital space telescope discovered candidate galaxies dating back to within three hundred million years of the Big Bang, challenging standard galactic formation timelines.",
         "Space observatory data revealed primordial galaxies formed only 300 million years following the Big Bang, prompting revisions to cosmological models of early galaxy evolution.",
         "Deep-field telescope observations identified primordial galaxy candidates formed within 300 million years of the Big Bang, prompting adjustments to early cosmological models."),

        ("High-Speed Rail Infrastructure",
         "Transportation authorities inaugurated the country's first high-speed rail corridor connecting the northern industrial hub with the capital city in under ninety minutes.",
         "The high-speed rail link connecting the capital with northern manufacturing regions opened for service, cutting transit times to under 90 minutes.",
         "A dedicated high-speed rail corridor commenced commercial operations, reducing travel duration between the northern manufacturing region and the capital to 90 minutes."),

        ("Urban Air Quality Regulations",
         "The environmental agency mandated stringent low-emission zones in central metropolitan districts, imposing daily congestion charges on older diesel passenger vehicles and heavy freight trucks.",
         "Environmental regulators established clean air zones in downtown urban centers, levying access fees on older diesel cars and commercial cargo trucks.",
         "Metropolitan environmental authorities established strict low-emission zones, imposing daily access charges on older diesel vehicles and commercial freight carriers."),

        ("Nuclear Fusion Energy Milestone",
         "Physicists at the national ignition facility achieved a net energy gain in a controlled laser fusion experiment, producing more fusion energy than the laser energy delivered to the target capsule.",
         "Scientists reported scientific energy breakeven in laser-driven fusion, generating more thermonuclear yield than the optical energy absorbed by the fuel target.",
         "National research scientists achieved scientific net energy gain in laser-driven fusion experiments, yielding greater thermal energy output than the laser input."),

        ("Cross-Border Financial Telecommunications",
         "Financial authorities announced the successful pilot of an interbank distributed ledger network for instantaneous settlement of cross-border wholesale payments with zero foreign exchange spread.",
         "Central banks completed a successful trial of a blockchain settlement network enabling real-time cross-border interbank transfers without currency conversion markups.",
         "An interbank distributed ledger platform completed trial testing, achieving instant settlement of cross-border financial transactions without currency conversion markups."),

        ("Wildfire Containment Emergency",
         "Emergency response personnel mobilized thousands of firefighters and aerial tankers to contain an uncontained wildfire that destroyed dozens of homes and forced widespread evacuations.",
         "Thousands of emergency responders and air tankers deployed to battle an out-of-control wildfire that consumed multiple residences and triggered mandatory evacuation notices.",
         "Emergency authorities dispatched aerial water bombers and thousands of personnel to battle a rapidly spreading wildfire that destroyed residential structures and forced mass evacuations."),

        ("Telecommunications Spectrum Auction",
         "The federal communications commission raised twenty-five billion dollars in a record-breaking auction of mid-band radio spectrum licenses to accelerate national fifth-generation mobile deployment.",
         "A record government auction of mid-band wireless frequencies generated $25 billion from mobile operators seeking to expand 5G network coverage.",
         "The national telecommunications regulator raised $25 billion through auctioning mid-band radio spectrum to speed nationwide 5G mobile network deployment."),

        ("Global Shipping Canal Bottleneck",
         "A maritime logistics crisis intensified as severe drought restricted vessel transit capacity through the international canal, causing extensive shipping delays and rising cargo freight rates.",
         "Logistical disruptions mounted as low water levels reduced ship crossings through the interoceanic canal, triggering freight rate increases and delivery backlogs.",
         "Low water levels in the interoceanic shipping canal restricted daily vessel passages, creating shipping bottlenecks and driving up global freight rates."),

        ("Public Health Antibiotic Resistance Alert",
         "Global health officials issued an urgent alert warning of rising multi-drug resistant bacterial strains in hospital environments, calling for stricter antibiotic stewardship and development of novel therapeutics.",
         "Health authorities warned of escalating antibiotic-resistant superbugs in clinical settings, urging enhanced antimicrobial governance and investment in new antibiotic pipelines.",
         "Public health authorities sounded alarms regarding the spread of multi-drug resistant bacterial pathogens in clinical facilities, calling for strict antimicrobial stewardship."),

        ("Renewable Hydrogen Pipeline Network",
         "Energy ministers signed a multilateral declaration to build a pan-continental green hydrogen pipeline network by 2035, connecting renewable electrolysis plants in the south with heavy industrial clusters in the north.",
         "Governments finalized a cross-border pact to construct a hydrogen pipeline grid by 2035, transporting green hydrogen from southern electrolysis facilities to northern factories.",
         "A multi-governmental accord was ratified to construct a cross-border green hydrogen pipeline network by 2035 connecting southern renewable hubs with northern industrial centers.")
    ]

    for idx, (topic, src, susp, adv) in enumerate(news_topics, 1):
        pairs.append({
            "pair_id": f"NEWS_{idx:03d}",
            "domain": "news",
            "benchmark": "SemEval-2022 Task 8",
            "topic": topic,
            "source_text": src,
            "suspect_text": susp,
            "paraphrases": {
                "static_paraphrase": susp,
                "adversarial_paraphrase": adv,
                "claude_populated_paraphrase": ""
            },
            "metadata": {"domain_type": "Journalistic News & Current Affairs", "topic": topic}
        })

    return pairs

def main():
    pairs = create_120_benchmark_pairs()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    # Save both 120 dataset and default benchmark dataset
    path_120 = os.path.join(data_dir, "benchmark_pairs_120.json")
    path_default = os.path.join(data_dir, "benchmark_pairs.json")
    
    with open(path_120, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
        
    with open(path_default, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
        
    print(f"Successfully generated {len(pairs)} benchmark document pairs across 5 domains.")
    print(f"Saved to: {path_120}")
    print(f"Saved to: {path_default}")
    
    # Print domain breakdown
    from collections import Counter
    domains = Counter(p["domain"] for p in pairs)
    for dom, count in domains.items():
        print(f"  - {dom:12}: {count} pairs")

if __name__ == "__main__":
    main()
