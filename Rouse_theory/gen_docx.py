from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

t = doc.add_heading('Properties of the Rouse Model', level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    'Based on: P. E. Rouse Jr., "A Theory of the Linear Viscoelastic Properties '
    'of Dilute Solutions of Coiling Polymers," J. Chem. Phys. 21(7), 1272\u20131280 (1953)')
run.italic = True
run.font.size = Pt(10)

# =========================================================
# DEFINITION OF THE ROUSE MODEL
# =========================================================
doc.add_heading('Definition of the Rouse Model', level=1)
doc.add_paragraph(
    'The Rouse model, proposed by Prince E. Rouse Jr. in 1953, is a molecular theory for '
    'the linear viscoelastic behaviour of dilute solutions of flexible, coiling polymer '
    'molecules. The model represents a polymer chain as a series of N identical submolecules '
    '(beads) connected by entropic springs, where each submolecule is long enough for its '
    'end-to-end separation to obey Gaussian statistics. The beads experience frictional drag '
    'from the surrounding solvent but do not interact hydrodynamically with one another '
    '(the "free-draining" assumption).')
doc.add_paragraph(
    'Under an applied velocity gradient the distribution of molecular configurations is '
    'perturbed away from equilibrium, storing free energy. Coordinated Brownian motions of '
    'the segments then drive the configurations back toward their most probable distribution. '
    'By decomposing the coupled segmental motions into a set of independent normal modes \u2014 '
    'each with its own characteristic relaxation time \u2014 the theory yields closed-form '
    'expressions for the complex viscosity, storage modulus, loss modulus, and steady-flow '
    'viscosity of the polymer solution in terms of the molecular weight, concentration, '
    'solvent viscosity, and temperature.')

# =========================================================
# SECTION I - STATIC PROPERTIES
# =========================================================
doc.add_heading('(I) Static Properties', level=1)
doc.add_paragraph(
    'The static (equilibrium) properties of the Rouse model describe the time-independent, '
    'structural characteristics of the polymer chain at thermodynamic equilibrium.')

doc.add_heading('1. Gaussian Sub-chain Model', level=2)
doc.add_paragraph(
    'The polymer molecule is divided into N equal submolecules (sub-chains). Each submolecule '
    'is a portion of polymer chain long enough that the separation of its ends obeys, to a '
    'first approximation, a Gaussian probability distribution. The configuration of the entire '
    'molecule is specified by the set of N end-to-end vectors of its submolecules.')

doc.add_heading('2. Gaussian Probability Distribution of a Submolecule', level=2)
doc.add_paragraph(
    'The probability that the end-to-end vector of a single submolecule falls in the '
    'volume element dx dy dz is:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03c8(x, y, z) dx dy dz = (\u03b2/\u03c0)^(3/2) exp[\u2212\u03b2(x\u00b2 + y\u00b2 + z\u00b2)] dx dy dz')
run.italic = True
doc.add_paragraph(
    'where \u03b2 = 3/(2\u03c3\u00b2) and \u03c3\u00b2 is the mean-square separation of the ends of a submolecule.')

doc.add_heading('3. Configuration Probability of the Entire Molecule', level=2)
doc.add_paragraph(
    'The probability of the entire molecule being in a particular configuration is the '
    'product of the individual submolecule probabilities:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    '\u03a8 d\u03c6\u1d62 = \u220fj=1..N \u03c8(xj, yj, zj) dxj dyj dzj = '
    '(\u03b2/\u03c0)^(3N/2) exp[\u2212\u03b2 \u03a3j (xj\u00b2 + yj\u00b2 + zj\u00b2)] d\u03c6\u1d62')
run.italic = True
doc.add_paragraph(
    'This assumes that the submolecule configurations are statistically independent \u2014 '
    'the group of submolecules with a given end-to-end separation are randomly distributed '
    'over all available configurations consistent with that separation.')

doc.add_heading('4. Mean-Square End-to-End Distance', level=2)
doc.add_paragraph('The mean-square end-to-end distance of the entire polymer molecule is:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('S\u00b2 = N\u03c3\u00b2')
run.italic = True
run.bold = True
doc.add_paragraph(
    'where N is the number of submolecules and \u03c3\u00b2 is the mean-square end-to-end distance '
    'of one submolecule. This is equivalent to a freely jointed chain of n links of length l, '
    'for which p(x,y,z) = (b\u00b2/\u03c0\u00bd) exp[\u2212b\u00b2(x\u00b2+y\u00b2+z\u00b2)] with b\u00b2 = 3/(2nl\u00b2).')

doc.add_heading('5. Thermodynamic Potential (Free Energy)', level=2)
doc.add_paragraph(
    'The Helmholtz free energy change is \u0394A = \u2212T\u0394S, where the entropy change upon '
    'perturbation by a velocity gradient \u03b1 is given by Wall\'s equation:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u0394S = k \u03a3\u1d62 s\u1d62 ln(n\u1d62/s\u1d62)')
run.italic = True
doc.add_paragraph('The thermodynamic potential of the polymer molecules is:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03bc = kT [1 + ln(s\u1d62/n\u1d62)]  \u2192  \u03bc = kT [1 + \u03b1f]')
run.italic = True
doc.add_paragraph(
    'where f is a function of the 3N coordinates, and the perturbation is small enough '
    'that only first-order terms in \u03b1 are retained.')

doc.add_heading('6. Connectivity Matrix A', level=2)
doc.add_paragraph(
    'The connectivity (Rouse matrix) A is a symmetric tridiagonal N\u00d7N matrix with elements '
    '2 on the diagonal and \u22121 on the sub/super-diagonals. It encodes the linear connectivity '
    'of the chain and governs both the equilibrium and dynamic behaviour through its eigenvalues:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03bbp = 4 sin\u00b2[p\u03c0 / 2(N+1)],   p = 1, 2, \u2026, N')
run.italic = True
run.bold = True

# =========================================================
# SECTION II - DYNAMIC PROPERTIES
# =========================================================
doc.add_heading('(II) Dynamic Properties', level=1)
doc.add_paragraph(
    'The dynamic properties describe the time-dependent, viscoelastic response of the '
    'Rouse chain to an applied velocity gradient (shear). The theory resolves the coordinated '
    'motions of the chain into a series of normal modes, each with its own relaxation time.')

doc.add_heading('1. Equations of Motion', level=2)
doc.add_paragraph(
    'The motion of the polymer molecule in the x-direction is governed by the matrix equation:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('x\u0307t = \u03b1z \u2212 BA{\u2207z \u03bc}')
run.italic = True
doc.add_paragraph(
    'where B is the mobility of the end of a submolecule (inversely proportional to the '
    'length of chain in the submolecule), and A is the Rouse connectivity matrix. Analogous '
    'equations hold for the y and z directions.')

doc.add_heading('2. Normal-Mode Transformation', level=2)
doc.add_paragraph(
    'An orthogonal transformation R diagonalises the Rouse matrix: R\u207b\u00b9AR = \u039b = [\u03bbp \u03b4pq]. '
    'In the transformed (normal) coordinates u, v, w the equations of motion decouple:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('u\u0307t = \u03b1w \u2212 B\u039b{\u2207u \u03bc}')
run.italic = True
doc.add_paragraph(
    'Each normal mode p represents a specific pattern of coordinated segmental motion '
    'along the chain.')

doc.add_heading('3. Relaxation Times', level=2)
doc.add_paragraph('Each mode p has a characteristic relaxation time:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03c4p = (4\u03b2Dp)\u207b\u00b9 = \u03c3\u00b2 [24BkT sin\u00b2(p\u03c0 / 2(N+1))]\u207b\u00b9,   p = 1, 2, \u2026, N')
run.italic = True
run.bold = True
doc.add_paragraph('For modes with p < N/5 (long-wavelength modes), this simplifies to:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03c4p \u2248 \u03c3\u00b2(N+1)\u00b2 / (6\u03c0\u00b2p\u00b2BkT)  \u2248  \u03c3\u00b2N\u00b2 / (6\u03c0\u00b2p\u00b2BkT)')
run.italic = True
doc.add_paragraph(
    'The longest relaxation time (p = 1) dominates the low-frequency viscoelastic response. '
    'The ratio between successive relaxation times is large for small p and approaches unity '
    'as p \u2192 N. The shortest relaxation time is \u03c4N \u2248 \u03c3\u00b2/(24BkT).')

doc.add_heading('4. Relaxation Time from Measurable Quantities', level=2)
doc.add_paragraph('The relaxation times can be expressed in terms of steady-flow viscosity data:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03c4p = 6(\u03b70 \u2212 \u03b7s) / (\u03c0\u00b2p\u00b2nkT)')
run.italic = True
run.bold = True
doc.add_paragraph(
    'where \u03b70 is the steady-flow viscosity of the solution, \u03b7s is the solvent viscosity, '
    'n is the number of molecules per unit volume, k is Boltzmann\'s constant, and T is the '
    'absolute temperature.')

doc.add_heading('5. Complex Viscosity  \u03b7* = \u03b71 \u2212 i\u03b72', level=2)
doc.add_paragraph('Under oscillatory shear at angular frequency \u03c9:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03b71 = \u03b7s + nkT \u03a3p=1..N  \u03c4p / (1 + \u03c9\u00b2\u03c4p\u00b2)')
run.italic = True
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03b72 = nkT \u03a3p=1..N  \u03c9\u03c4p\u00b2 / (1 + \u03c9\u00b2\u03c4p\u00b2)')
run.italic = True
doc.add_paragraph(
    '\u03b71 is the real (in-phase) part of the complex viscosity; \u03b72 is the imaginary '
    '(out-of-phase) part. At low frequencies (\u03c9\u03c41 \u226a 1), \u03b71 \u2192 \u03b70 and \u03b72 \u2192 0. '
    'At high frequencies, the polymer contribution vanishes and \u03b71 \u2192 \u03b7s.')

doc.add_heading('6. Steady-Flow (Zero-Shear) Viscosity', level=2)
doc.add_paragraph('Setting \u03c9 = 0 yields the steady-flow viscosity:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03b70 = \u03b7s + n\u03c3\u00b2N(N+2) / (36B)  \u2248  \u03b7s + nNS\u00b2 / (36B)')
run.italic = True
run.bold = True
doc.add_paragraph(
    'This expression is analogous to Debye\'s result for "free-draining" chains, '
    'with the number of atomic groups replaced by N (number of submolecules) and the '
    'frictional coefficient replaced by B\u207b\u00b9 (the reciprocal of the mobility).')

doc.add_heading('7. Complex Shear Modulus  G* = G\u2081 + iG\u2082', level=2)
doc.add_paragraph('The storage and loss moduli are:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('G\u2081 (storage modulus) = nkT \u03a3p=1..N  \u03c9\u00b2\u03c4p\u00b2 / (1 + \u03c9\u00b2\u03c4p\u00b2)')
run.italic = True
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('G\u2082 (loss modulus) = \u03c9\u03b7s + nkT \u03a3p=1..N  \u03c9\u03c4p / (1 + \u03c9\u00b2\u03c4p\u00b2)')
run.italic = True
doc.add_paragraph(
    'Each relaxation mode contributes nkT to G\u2081 at high frequencies, behaving as a '
    '"generalized Maxwell model." The longer relaxation times account for practically all '
    'of the viscosity; the shorter times, though many in number, contribute only a small '
    'fraction of the total viscosity.')

doc.add_heading('8. Simplified High-Frequency / Low-Frequency Expressions', level=2)
doc.add_paragraph('For intermediate-to-high frequencies (2 < \u03c9\u03c41 < N\u00b2/250):')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\u03b71 \u2248 \u03b7s + 3(\u03b70 \u2212 \u03b7s) / [\u03c0(2\u03c9\u03c41)\u00bd]')
run.italic = True
doc.add_paragraph('For 5 < \u03c9\u03c41 < N\u00b2/250:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('G\u2081 \u2248 nkT [\u03c0/2 \u00b7 (\u03c9\u03c41/2)\u00bd \u2212 0.5]')
run.italic = True
doc.add_paragraph(
    'At high frequencies \u03b72 approaches (\u03b71 \u2212 \u03b7s), so the polymer contribution to '
    'the loss modulus approaches the polymer contribution to the viscous dissipation.')

doc.add_heading('9. Dependence of Relaxation Times on Physical Parameters', level=2)
doc.add_paragraph('The relaxation times depend on three factors:')
items = [
    'Inversely proportional to the absolute temperature T.',
    'Inversely proportional to the mobility B of the submolecule end (which decreases '
    'as the viscous forces from the surrounding medium increase).',
    'Directly proportional to \u03c3\u00b2 (the mean-square separation of the submolecule ends), '
    'and consequently to S\u00b2 (the mean-square end-to-end distance of the entire molecule). '
    'Changes in temperature cause changes in the extension of dissolved polymer molecules, '
    'which are reflected in the intrinsic viscosity \u2014 consistent with the concept of '
    '"reduced variables" introduced by Ferry.',
]
for item in items:
    doc.add_paragraph(item, style='List Bullet')

doc.add_heading('10. Viscous Dissipation and Energy Balance', level=2)
doc.add_paragraph(
    'The complex viscosity is obtained by equating the rate of input of free energy into '
    'the polymer molecules to the rate of work done on each molecule as it moves with the '
    'velocity of the surrounding liquid. The average rate per molecule of doing work on the '
    'representative points in the 3N-dimensional configuration space is given by the scalar '
    'product of the liquid velocity Vt and the thermodynamic-potential gradient \u2207\u03bc.')

# Limitations
doc.add_heading('Limitations of the Theory', level=1)
items = [
    'Very rapid relaxation processes involving segments shorter than the submolecule are excluded.',
    'The obstruction of the motion of a segment by other segments with which it happens '
    'to be in contact is not taken into account (no hydrodynamic interaction).',
    'Polydispersity of any actual polymer is not accounted for.',
    'Intramolecular and intermolecular interferences on the motions of polymer segments '
    'are neglected \u2014 the theory is therefore expected to be satisfactory only for dilute solutions.',
]
for item in items:
    doc.add_paragraph(item, style='List Bullet')

# =========================================================
# SECTION III - APPLICABILITY TO PROTEINS
# =========================================================
doc.add_heading('(III) Applicability of the Rouse Model to Proteins', level=1)

doc.add_paragraph(
    'Proteins are linear heteropolymers \u2014 polypeptide chains built from a sequence of '
    'amino acid residues joined by peptide bonds. Because the Rouse model was developed '
    'for homopolymer chains in dilute solution, its applicability to proteins is conditional '
    'and depends strongly on the conformational state of the protein.')

doc.add_heading('Where the Rouse Model Applies', level=2)

doc.add_heading('Intrinsically Disordered Proteins (IDPs)', level=3)
doc.add_paragraph(
    'IDPs lack a stable three-dimensional fold and sample a large ensemble of rapidly '
    'interconverting conformations in solution. In this respect they resemble the flexible, '
    'coiling polymers for which the Rouse model was designed. Experimental and computational '
    'studies have shown that the internal dynamics of IDPs \u2014 particularly the distance '
    'autocorrelation functions and reconfiguration times measured by single-molecule FRET '
    'and fluorescence correlation spectroscopy \u2014 can be well described by Rouse-like '
    'normal modes. The scaling of the longest reconfiguration time with chain length '
    '(\u03c4 ~ N\u00b2) and the mode-number dependence (\u03c4p ~ 1/p\u00b2) are characteristic '
    'Rouse signatures that have been observed in unstructured polypeptides.')

doc.add_heading('Unfolded / Denatured Proteins', level=3)
doc.add_paragraph(
    'Under strongly denaturing conditions (high concentrations of urea or guanidinium '
    'chloride, extreme pH, or high temperature), globular proteins lose their native '
    'tertiary and secondary structure and behave as expanded, disordered chains. In this '
    'unfolded state the chain dynamics are again amenable to a Rouse description. NMR '
    'relaxation measurements on denatured proteins have revealed backbone motions whose '
    'correlation times follow the mode-dependent scaling predicted by the Rouse model. '
    'The model provides a useful baseline: deviations from Rouse behaviour in a nominally '
    'unfolded protein indicate residual structure or non-random interactions.')

doc.add_heading('Disordered Linkers and Loops', level=3)
doc.add_paragraph(
    'Multi-domain proteins often contain flexible, disordered linker regions connecting '
    'folded domains. The dynamics of these linker segments \u2014 typically tens of residues '
    'long \u2014 can be modelled as short Rouse chains tethered at both ends. Similarly, long '
    'surface loops that are not stabilised by tertiary contacts may exhibit Rouse-like '
    'fluctuations on nanosecond-to-microsecond timescales.')

doc.add_heading('Where the Rouse Model Breaks Down for Proteins', level=2)

doc.add_heading('Folded (Native-State) Globular Proteins', level=3)
doc.add_paragraph(
    'A folded protein is a compact, structured object stabilised by a dense network of '
    'hydrogen bonds, hydrophobic contacts, salt bridges, and disulphide bonds. Its internal '
    'dynamics are dominated by collective elastic vibrations around a well-defined '
    'equilibrium structure, not by the diffusive, entropy-driven rearrangements of a '
    'flexible chain. The fundamental Rouse assumptions \u2014 Gaussian sub-chain statistics, '
    'statistical independence of sub-chain configurations, and absence of intra-chain '
    'contacts \u2014 are all violated in a folded protein. Consequently, the Rouse model '
    'is not applicable to the native state of globular proteins.')

doc.add_heading('Hydrodynamic Interactions', level=3)
doc.add_paragraph(
    'The Rouse model assumes "free-draining" chains: each segment experiences friction '
    'with the solvent independently. In reality, the motion of one segment perturbs the '
    'solvent velocity field and thereby affects the drag on other segments (hydrodynamic '
    'interaction). For synthetic polymers this limitation is addressed by the Zimm model '
    '(1956), which incorporates a pre-averaged Oseen tensor. For proteins, hydrodynamic '
    'interactions can be significant, especially in compact or partially collapsed states. '
    'The Zimm model predicts a different scaling of the longest relaxation time with chain '
    'length (\u03c4 ~ N^(3\u03bd), where \u03bd is the Flory exponent) compared with the Rouse '
    'prediction (\u03c4 ~ N^(1+2\u03bd)). Experiments on unfolded proteins in good-solvent '
    'conditions often show Zimm-like rather than Rouse-like scaling, indicating that '
    'hydrodynamic interactions are not fully screened.')

doc.add_heading('Sequence Heterogeneity and Specific Interactions', level=3)
doc.add_paragraph(
    'Unlike the homopolymer assumed by Rouse, a protein has a specific amino acid sequence. '
    'Different residues have different sizes, charges, and hydrophobicities, leading to '
    'position-dependent friction coefficients, non-uniform stiffness along the backbone, '
    'and sequence-specific attractive or repulsive interactions. These effects can cause '
    'local compaction, transient secondary structure formation, or long-range contacts that '
    'break the assumptions of uniform Gaussian sub-chains. Generalised Rouse-type models '
    'with position-dependent spring constants or friction coefficients have been developed '
    'to partially address this heterogeneity.')

doc.add_heading('Excluded Volume Effects', level=3)
doc.add_paragraph(
    'The Rouse model treats the chain as a phantom \u2014 different parts of the chain can '
    'freely pass through one another. Real polypeptide chains exhibit excluded volume: two '
    'residues cannot occupy the same space. In good-solvent conditions the chain swells '
    'beyond the ideal Gaussian dimensions (S\u00b2 ~ N^(2\u03bd) with \u03bd \u2248 0.588 rather than '
    '\u03bd = 0.5). This changes both the equilibrium statistics and the mode-dependent '
    'relaxation spectrum relative to the ideal Rouse predictions.')

doc.add_heading('Summary: When to Use the Rouse Model for Proteins', level=2)
table = doc.add_table(rows=6, cols=2, style='Light Shading Accent 1')
table.cell(0, 0).text = 'Protein State'
table.cell(0, 1).text = 'Rouse Model Applicable?'
rows_data = [
    ('Intrinsically disordered proteins (IDPs)', 'Yes \u2014 good first approximation'),
    ('Fully denatured / unfolded proteins', 'Yes \u2014 useful baseline; deviations reveal residual structure'),
    ('Disordered linkers and long loops', 'Yes \u2014 for the disordered segment, with tethering corrections'),
    ('Folded globular proteins (native state)', 'No \u2014 assumptions fundamentally violated'),
    ('Partially folded / molten globule', 'Limited \u2014 may apply to disordered regions only'),
]
for i, (state, answer) in enumerate(rows_data, start=1):
    table.cell(i, 0).text = state
    table.cell(i, 1).text = answer

doc.add_paragraph('')  # spacing after table
doc.add_paragraph(
    'In conclusion, the Rouse model provides a valuable theoretical framework for '
    'understanding the dynamics of proteins in their unstructured states \u2014 whether '
    'intrinsically disordered, chemically denatured, or locally unfolded. For these '
    'systems it offers physically grounded predictions of reconfiguration times, internal '
    'relaxation spectra, and viscoelastic response. However, its assumptions of Gaussian '
    'statistics, uniform chain properties, and free-draining hydrodynamics are too '
    'restrictive for folded proteins, and extensions such as the Zimm model or '
    'sequence-specific generalisations are often needed for quantitative accuracy even '
    'in disordered polypeptides.')

# =========================================================
# SECTION IV - ROUSE MODEL AND THE SURPASS-ALPHA CG MODEL
# =========================================================
doc.add_heading('(IV) Applicability of the Rouse Model to the SURPASS-\u03b1 '
                'Coarse-Grained Model of Proteins', level=1)

doc.add_paragraph(
    'SURPASS-\u03b1 (Single United Residue per Pre-Averaged Secondary Structure '
    'fragment \u2014 alpha-carbon variant) is a coarse-grained simulation framework for '
    'proteins implemented in C#/.NET. It represents each amino acid residue by one or two '
    'pseudoatoms: a C\u03b1 bead on the backbone and an optional side-group (SG) bead. '
    'The beads are placed on an integer coordinate grid (for computational efficiency) '
    'and sampled by Metropolis Monte Carlo with pivot and tail rotation moves. The force '
    'field includes harmonic backbone bonds (CA\u2013CA), CA\u2013SG tether bonds, excluded-volume '
    'repulsion, Lennard-Jones and contact potentials, geometry-dependent hydrogen bonds, '
    'sequence-specific (Miyazawa\u2013Jernigan) contact energies, and aromatic stacking terms. '
    'Periodic boundary conditions with the minimum image convention are used throughout.')

doc.add_paragraph(
    'This section examines, feature by feature, where the Rouse model and SURPASS-\u03b1 '
    'share common ground and where they diverge, and assesses how and why Rouse-type '
    'analysis may (or may not) be meaningfully applied to SURPASS-\u03b1 simulations.')

# --- Structural parallels ---
doc.add_heading('Structural Parallels Between the Rouse Model and SURPASS-\u03b1', level=2)

doc.add_heading('1. Linear Bead\u2013Spring Chain Topology', level=3)
doc.add_paragraph(
    'Both models represent a polymer as a linear sequence of beads connected by springs. '
    'In Rouse the chain consists of N submolecules joined by entropic (Gaussian) springs. '
    'In SURPASS-\u03b1 the backbone is a sequence of C\u03b1 beads joined by harmonic bonds with '
    'equilibrium distance r\u2080 \u2248 3.8 \u00c5 and spring constant k. The connectivity matrix in '
    'SURPASS-\u03b1 is therefore identical in form to the Rouse tridiagonal matrix A (with '
    'elements 2 on the diagonal and \u22121 on the off-diagonals), provided the harmonic '
    'spring constant is uniform along the chain. This shared topology means that the '
    'normal-mode decomposition central to Rouse theory can, in principle, be directly '
    'applied to the SURPASS-\u03b1 backbone.')

doc.add_heading('2. No Explicit Angle or Dihedral Constraints', level=3)
doc.add_paragraph(
    'SURPASS-\u03b1 does not impose explicit bond-angle or dihedral-angle potentials on the '
    'backbone. Chain stiffness arises implicitly from the interplay of contact, '
    'hydrogen-bond, and excluded-volume energies rather than from explicit bending terms. '
    'The Rouse model likewise has no angular potentials \u2014 sub-chains are freely jointed. '
    'In regions of the SURPASS-\u03b1 chain where non-bonded interactions are weak (e.g. '
    'disordered loops or denatured segments), the effective backbone flexibility approaches '
    'the Rouse ideal of freely jointed sub-chains.')

doc.add_heading('3. Absence of Explicit Hydrodynamic Interactions', level=3)
doc.add_paragraph(
    'Neither model includes hydrodynamic interactions (HI). Rouse explicitly assumes '
    '"free-draining" chains in which each bead experiences independent frictional drag '
    'from the solvent. SURPASS-\u03b1 uses Monte Carlo sampling, which has no solvent velocity '
    'field at all \u2014 the solvent enters only implicitly through the effective potentials and '
    'the thermal energy kT in the Metropolis criterion. Because MC does not propagate '
    'hydrodynamic correlations, the dynamics sampled by SURPASS-\u03b1 are inherently '
    'free-draining, making the Rouse (rather than the Zimm) framework the natural '
    'theoretical reference for interpreting SURPASS-\u03b1 dynamical observables.')

# --- Key differences ---
doc.add_heading('Key Differences Between the Rouse Model and SURPASS-\u03b1', level=2)

doc.add_heading('1. Excluded Volume', level=3)
doc.add_paragraph(
    'The Rouse model treats the chain as a phantom: different parts of the chain may '
    'freely overlap. SURPASS-\u03b1 enforces excluded volume via a hard repulsive penalty '
    'when any two beads approach closer than a cutoff distance. Excluded volume causes '
    'the chain to swell beyond Gaussian dimensions and modifies the relaxation spectrum '
    '(the eigenvalue spacing of the effective connectivity matrix). In an extended, '
    'disordered chain the effect is moderate and can be absorbed into a renormalised '
    'effective bond length, preserving Rouse-like mode structure. In a compact globular '
    'state, excluded volume and dense packing dominate, and the Rouse picture breaks down.')

doc.add_heading('2. Sequence-Specific Interactions', level=3)
doc.add_paragraph(
    'Rouse assumes a homopolymer with identical spring constants and friction coefficients '
    'along the chain. SURPASS-\u03b1 is explicitly heteropolymeric: it classifies the 20 amino '
    'acid types into groups (small, aliphatic, aromatic, polar, basic, acidic, proline) and '
    'assigns sequence-specific pairwise contact energies via a 20\u00d720 Miyazawa\u2013Jernigan '
    'matrix. This heterogeneity means that (a) the effective spring constant between '
    'residues varies along the chain, (b) some residue pairs attract strongly while others '
    'repel, and (c) the chain may locally collapse or expand depending on its sequence. '
    'These effects break the uniform-chain assumption of Rouse. However, a generalised '
    'Rouse model with position-dependent friction coefficients and spring constants can '
    'partially accommodate this heterogeneity.')

doc.add_heading('3. Hydrogen Bonds and Secondary Structure', level=3)
doc.add_paragraph(
    'SURPASS-\u03b1 includes a detailed, geometry-dependent hydrogen-bond potential that '
    'stabilises \u03b1-helices (i, i+4 contacts) and \u03b2-sheets (long-range backbone contacts). '
    'These specific, directional interactions create stiff, correlated segments along the '
    'chain that are fundamentally different from the freely jointed sub-chains of Rouse. '
    'In a helix, several consecutive residues move as a rigid body; in a \u03b2-sheet, distant '
    'segments are locked together by a ladder of hydrogen bonds. Neither situation is '
    'described by independent Gaussian springs. The Rouse normal modes assume that the '
    'only coupling between beads is through nearest-neighbour springs \u2014 hydrogen bonds '
    'introduce strong non-nearest-neighbour couplings that fundamentally alter the '
    'eigenvalue spectrum.')

doc.add_heading('4. Lennard-Jones and Contact Potentials', level=3)
doc.add_paragraph(
    'Beyond excluded volume, SURPASS-\u03b1 includes attractive Lennard-Jones and multi-zone '
    'contact potentials between non-bonded beads. These drive chain compaction and the '
    'formation of a hydrophobic core. In the Rouse model, non-bonded interactions are '
    'entirely absent \u2014 the only forces are the entropic springs connecting consecutive '
    'beads. The presence of strong non-bonded attractions in SURPASS-\u03b1 means that at '
    'low temperatures (below the folding temperature) the chain collapses into a compact '
    'globule whose internal dynamics are nothing like Rouse diffusion.')

doc.add_heading('5. Side-Group Beads', level=3)
doc.add_paragraph(
    'SURPASS-\u03b1 optionally attaches a side-group (SG) bead to each C\u03b1, connected by a '
    'harmonic tether. This gives each residue an internal degree of freedom absent in the '
    'Rouse model, where each submolecule is a structureless point. Side-group motions '
    '(including aromatic stacking between Phe, Tyr, and Trp residues) introduce additional '
    'relaxation modes that have no Rouse counterpart.')

doc.add_heading('6. Monte Carlo vs. Brownian Dynamics', level=3)
doc.add_paragraph(
    'The Rouse model describes continuous Brownian dynamics governed by Langevin-type '
    'equations of motion, producing a well-defined time axis with physical units. '
    'SURPASS-\u03b1 uses Metropolis Monte Carlo with pivot and tail rotation moves. MC '
    'sweeps do not correspond to physical time in any simple way; the acceptance rate, '
    'move amplitude, and move type all affect the effective timescale. While equilibrium '
    '(static) properties sampled by MC can be directly compared with Rouse equilibrium '
    'predictions, dynamic quantities (relaxation times, diffusion coefficients, viscosity) '
    'require a mapping between MC sweeps and physical time \u2014 typically via calibration '
    'against a known relaxation process.')

# --- When Rouse analysis applies to SURPASS-alpha ---
doc.add_heading('When and How Rouse Analysis Can Be Applied to SURPASS-\u03b1 Simulations', level=2)

doc.add_heading('Applicable Scenarios', level=3)
items = [
    ('High-temperature / unfolded ensembles. ',
     'At temperatures well above the folding temperature, the hydrogen-bond and contact '
     'energies are thermally overwhelmed and the SURPASS-\u03b1 chain behaves as a '
     'self-avoiding bead\u2013spring chain. In this regime the backbone dynamics are '
     'well approximated by a Rouse model with excluded-volume corrections. The '
     'normal-mode autocorrelation functions of the C\u03b1 coordinates should decay '
     'exponentially with mode-dependent relaxation times scaling as \u03c4\u209a ~ 1/p\u00b2 '
     '(or ~ 1/p^(2+2\u03bd\u22121) with excluded volume).'),
    ('Intrinsically disordered sequences. ',
     'If the amino acid sequence lacks strong hydrophobic or H-bonding propensity '
     '(e.g. polyglycine, poly-serine, or realistic IDP sequences), the SURPASS-\u03b1 '
     'chain remains expanded and disordered even at moderate temperatures. The Rouse '
     'model then provides a natural reference for the segmental dynamics, and deviations '
     'from Rouse scaling can be used to quantify residual intra-chain correlations.'),
    ('Normal-mode analysis of backbone fluctuations. ',
     'Regardless of folding state, one can always decompose the C\u03b1 coordinate '
     'trajectory from a SURPASS-\u03b1 simulation into normal modes by applying the '
     'orthogonal transformation R that diagonalises the tridiagonal connectivity '
     'matrix A. The resulting mode amplitudes and autocorrelation functions can then '
     'be compared with Rouse predictions. For folded proteins the low-order modes will '
     'deviate strongly (stiffened by tertiary contacts), but the high-order modes '
     '(local fluctuations) may still follow Rouse-like statistics.'),
    ('Equilibrium chain statistics. ',
     'Static properties such as the mean-square internal distances \u27e8(R\u1d62 \u2212 R\u2c7c)\u00b2\u27e9 '
     'as a function of sequence separation |i \u2212 j| can be extracted from SURPASS-\u03b1 '
     'trajectories and compared with the Rouse prediction \u27e8R\u00b2\u27e9 = |i \u2212 j| \u03c3\u00b2. '
     'The ratio of observed to predicted values quantifies the deviation from ideal '
     'chain behaviour.'),
]
for title, body in items:
    p = doc.add_paragraph()
    run = p.add_run(title)
    run.bold = True
    p.add_run(body)

doc.add_heading('Non-Applicable Scenarios', level=3)
items_no = [
    ('Folded native states. ',
     'When SURPASS-\u03b1 is used to study the folded ensemble of a globular protein, '
     'the dense network of hydrogen bonds, hydrophobic contacts, and excluded-volume '
     'constraints renders the Rouse description invalid. The effective connectivity '
     'is no longer tridiagonal; it includes all residue pairs in contact.'),
    ('Folding transitions. ',
     'During a folding or unfolding transition the chain passes through partially '
     'structured intermediates with heterogeneous local dynamics. The Rouse assumption '
     'of uniform, independent sub-chains does not hold in this regime.'),
    ('Quantitative dynamical predictions. ',
     'Because SURPASS-\u03b1 uses Monte Carlo rather than Brownian dynamics, absolute '
     'relaxation times cannot be directly extracted without an external calibration. '
     'Rouse theory gives \u03c4\u209a in physical units (seconds) via the bead friction '
     'coefficient; MC sweeps have no intrinsic timescale.'),
]
for title, body in items_no:
    p = doc.add_paragraph()
    run = p.add_run(title)
    run.bold = True
    p.add_run(body)

# --- Summary comparison table ---
doc.add_heading('Summary Comparison', level=2)
table2 = doc.add_table(rows=10, cols=3, style='Light Shading Accent 1')
headers = ['Feature', 'Rouse Model', 'SURPASS-\u03b1']
for j, h in enumerate(headers):
    table2.cell(0, j).text = h

comparison = [
    ('Chain topology', 'Linear bead\u2013spring', 'Linear C\u03b1 bead\u2013spring (+ optional SG)'),
    ('Bond potential', 'Gaussian / entropic spring', 'Harmonic spring (E = \u00bdk(r\u2212r\u2080)\u00b2)'),
    ('Angle / dihedral terms', 'None (freely jointed)', 'None explicit; implicit via potentials'),
    ('Non-bonded interactions', 'None (phantom chain)', 'Excluded volume, LJ, contacts, H-bonds, '
     'aromatic stacking, MJ matrix'),
    ('Sequence specificity', 'Homopolymer', 'Heteropolymer (20 AA types, 8 CG classes)'),
    ('Hydrodynamic interactions', 'None (free-draining)', 'None (MC has no solvent flow)'),
    ('Dynamics', 'Brownian / Langevin (continuous time)', 'Metropolis MC (discrete sweeps)'),
    ('Normal modes', 'Analytical eigenvalues \u03bb\u209a = 4 sin\u00b2(p\u03c0/2(N+1))',
     'Same eigenvalues apply to bare backbone; '
     'modified by non-bonded couplings'),
    ('Applicable regime', 'Dilute, flexible, unstructured chains', 'Unfolded / disordered chains '
     'at high T; breaks down for folded states'),
]
for i, (feat, rouse, surpass) in enumerate(comparison, start=1):
    table2.cell(i, 0).text = feat
    table2.cell(i, 1).text = rouse
    table2.cell(i, 2).text = surpass

doc.add_paragraph('')

# --- Concluding remarks ---
doc.add_heading('Concluding Remarks', level=2)
doc.add_paragraph(
    'The SURPASS-\u03b1 coarse-grained model shares the most fundamental structural feature '
    'of the Rouse model \u2014 a linear chain of beads connected by harmonic springs with no '
    'explicit angular constraints and no hydrodynamic interactions. This makes the Rouse '
    'normal-mode framework a natural and informative analytical tool for SURPASS-\u03b1 '
    'simulations, particularly for unfolded, denatured, or intrinsically disordered '
    'protein chains at elevated temperatures. In these regimes, Rouse theory provides '
    'quantitative predictions for mode-dependent relaxation, internal distance scaling, '
    'and chain reconfiguration dynamics that can be directly tested against SURPASS-\u03b1 '
    'trajectories.')
doc.add_paragraph(
    'However, the rich energy landscape of SURPASS-\u03b1 \u2014 including sequence-specific '
    'contacts, hydrogen bonds, aromatic stacking, and excluded volume \u2014 introduces '
    'strong deviations from Rouse behaviour whenever these interactions drive the chain '
    'toward structured or compact conformations. In such cases the Rouse model serves '
    'best as a null hypothesis: departures from Rouse scaling in the SURPASS-\u03b1 '
    'normal-mode spectrum directly quantify the effect of specific interactions and '
    'structural organisation on the chain dynamics. This makes the Rouse\u2013SURPASS-\u03b1 '
    'comparison a powerful diagnostic tool for distinguishing between ideal-chain '
    'behaviour and the emergence of protein-like structural features in coarse-grained '
    'simulations.')

doc.save(r'P:\Research Papers\Rouse_theory\Rouse_Model_Properties.docx')
print('Document saved successfully.')
