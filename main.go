package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ===================== Dataset de exemplo =====================

type Sample struct {
	Text  string `json:"text"`
	Label string `json:"intent"`
}

// ===================== Tokenização & n-grams =====================

var splitRe = regexp.MustCompile(`[^\p{L}\p{N}]+`)

func tokenize(s string) []string {
	s = strings.ToLower(s)
	parts := splitRe.Split(s, -1)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// 1-gram + 2-grams
func ngrams(tokens []string) []string {
	if len(tokens) == 0 {
		return nil
	}
	out := make([]string, 0, len(tokens)*2)
	out = append(out, tokens...) // unigrams
	for i := 0; i+1 < len(tokens); i++ {
		out = append(out, tokens[i]+"_"+tokens[i+1])
	}
	return out
}

// ===================== Vocabulário + TF-IDF =====================

type Vocab struct {
	Index map[string]int `json:"index"`
	Words []string       `json:"words"`
	IDF   []float32      `json:"idf"`
}

func buildVocabTFIDF(texts []string, minDF int) Vocab {
	df := map[string]int{}
	for _, t := range texts {
		toks := ngrams(tokenize(t))
		seen := map[string]bool{}
		for _, tok := range toks {
			if !seen[tok] {
				df[tok]++
				seen[tok] = true
			}
		}
	}
	words := make([]string, 0, len(df))
	for w, c := range df {
		if c >= minDF {
			words = append(words, w)
		}
	}
	sort.Strings(words)
	index := make(map[string]int, len(words))
	for i, w := range words {
		index[w] = i
	}
	N := float64(len(texts))
	idf := make([]float32, len(words))
	for i, w := range words {
		d := float64(df[w])
		idf[i] = float32(math.Log((N+1.0)/(d+1.0)) + 1.0)
	}
	return Vocab{Index: index, Words: words, IDF: idf}
}

// Vetor esparso (índices + valores)
type SparseVec struct {
	Idx []int
	Val []float32
}

func vectorizeTFIDF_Sparse(v Vocab, text string, l2normalize bool) SparseVec {
	counts := map[int]float32{}
	for _, tok := range ngrams(tokenize(text)) {
		if j, ok := v.Index[tok]; ok {
			counts[j] += 1.0
		}
	}
	if len(counts) == 0 {
		return SparseVec{}
	}
	// aplica IDF
	var idx []int
	var val []float32
	for j, tf := range counts {
		x := tf * v.IDF[j]
		idx = append(idx, j)
		val = append(val, x)
	}
	// normalização L2
	if l2normalize {
		var norm float32
		for _, x := range val {
			norm += x * x
		}
		if norm > 0 {
			n := float32(math.Sqrt(float64(norm)))
			for i := range val {
				val[i] /= n
			}
		}
	}
	// manter índices ordenados (opcional)
	sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
	// reordenar valores conforme idx
	tmp := make([]float32, len(val))
	copy(tmp, val)
	pos := make(map[int]int, len(idx))
	for k, j := range idx {
		pos[j] = k
	}
	for j, tf := range counts {
		k := sort.SearchInts(idx, j)
		tmp[k] = tf * v.IDF[j] // já normalizado acima
	}
	return SparseVec{Idx: idx, Val: val}
}

// ===================== Split & métricas =====================

func trainTestSplit(samples []Sample, testRatio float64, seed int64) (train, test []Sample) {
	r := rand.New(rand.NewSource(seed))
	shuffled := make([]Sample, len(samples))
	copy(shuffled, samples)
	r.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	nTest := int(float64(len(shuffled)) * testRatio)
	test = shuffled[:nTest]
	train = shuffled[nTest:]
	return
}

func uniqueLabels(samples []Sample) []string {
	seen := map[string]bool{}
	for _, s := range samples {
		seen[s.Label] = true
	}
	var labels []string
	for k := range seen {
		labels = append(labels, k)
	}
	sort.Strings(labels)
	return labels
}

func accuracy(yTrue, yPred []string) float64 {
	if len(yTrue) != len(yPred) {
		log.Fatal("tamanhos incompatíveis")
	}
	c := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			c++
		}
	}
	return float64(c) / float64(len(yTrue))
}

// ===================== Regressão Logística Multiclasse (Softmax) =====================

type SoftmaxModel struct {
	Labels []string    `json:"labels"`
	W      [][]float32 `json:"W"` // [C][D]
	B      []float32   `json:"B"` // [C]
}

func newSoftmax(labels []string, dim int, seed int64) *SoftmaxModel {
	r := rand.New(rand.NewSource(seed))
	W := make([][]float32, len(labels))
	for c := range W {
		W[c] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			W[c][j] = float32((r.Float64()*2 - 1) * 0.01)
		}
	}
	B := make([]float32, len(labels))
	return &SoftmaxModel{Labels: labels, W: W, B: B}
}

func dotSparse(w []float32, x SparseVec) float32 {
	var s float32
	for k, j := range x.Idx {
		s += w[j] * x.Val[k]
	}
	return s
}

func (m *SoftmaxModel) predictOne(x SparseVec) string {
	if len(m.Labels) == 0 {
		return ""
	}
	C := len(m.Labels)
	logits := make([]float32, C)
	var maxLogit float32 = -1e30
	for c := 0; c < C; c++ {
		z := dotSparse(m.W[c], x) + m.B[c]
		logits[c] = z
		if z > maxLogit {
			maxLogit = z
		}
	}
	// softmax (estável)
	var sum float32
	for c := 0; c < C; c++ {
		logits[c] = float32(math.Exp(float64(logits[c] - maxLogit)))
		sum += logits[c]
	}
	best := 0
	bestVal := float32(-1)
	for c := 0; c < C; c++ {
		p := logits[c] / sum
		if p > bestVal {
			bestVal = p
			best = c
		}
	}
	return m.Labels[best]
}

// Treino SGD mini-batch com weight decay (L2)
type TrainCfg struct {
	LR          float32 // learning rate
	L2          float32 // weight decay
	Epochs      int
	BatchSize   int
	Patience    int // early stopping (épocas sem melhora)
	ValSplit    float64
	Seed        int64
	L2Normalize bool
}

func trainSoftmax(cfg TrainCfg, labels []string, vocab Vocab, train []Sample) (*SoftmaxModel, float64) {
	// split train/val
	r := rand.New(rand.NewSource(cfg.Seed))
	shuf := make([]Sample, len(train))
	copy(shuf, train)
	r.Shuffle(len(shuf), func(i, j int) { shuf[i], shuf[j] = shuf[j], shuf[i] })
	nVal := int(float64(len(shuf)) * cfg.ValSplit)
	val := shuf[:nVal]
	tr := shuf[nVal:]

	// vetorização esparsa
	X := make([]SparseVec, len(tr))
	Y := make([]int, len(tr))
	Lmap := map[string]int{}
	for i, lb := range labels {
		Lmap[lb] = i
	}
	for i, s := range tr {
		X[i] = vectorizeTFIDF_Sparse(vocab, s.Text, cfg.L2Normalize)
		Y[i] = Lmap[s.Label]
	}
	Xv := make([]SparseVec, len(val))
	Yv := make([]int, len(val))
	for i, s := range val {
		Xv[i] = vectorizeTFIDF_Sparse(vocab, s.Text, cfg.L2Normalize)
		Yv[i] = Lmap[s.Label]
	}

	// C := len(labels)
	D := len(vocab.Words)
	m := newSoftmax(labels, D, cfg.Seed)

	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}

	bestValAcc := -1.0
	best := *m
	noImprove := 0

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		r.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		for start := 0; start < len(indices); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(indices) {
				end = len(indices)
			}

			// Weight decay (aplicado só nos pesos tocados)
			batch := indices[start:end]
			for _, i := range batch {
				x := X[i]
				y := Y[i]

				// logits
				Cs := len(m.Labels)
				logits := make([]float32, Cs)
				var maxz float32 = -1e30
				for c := 0; c < Cs; c++ {
					z := dotSparse(m.W[c], x) + m.B[c]
					logits[c] = z
					if z > maxz {
						maxz = z
					}
				}
				// softmax
				var sum float32
				for c := 0; c < Cs; c++ {
					logits[c] = float32(math.Exp(float64(logits[c] - maxz)))
					sum += logits[c]
				}
				// Gradiente e atualização por amostra (SGD)
				for c := 0; c < Cs; c++ {
					p := logits[c] / sum
					grad := p
					if c == y {
						grad -= 1.0
					}
					// bias
					m.B[c] -= cfg.LR * grad
					// pesos esparsos + L2 local
					for k, j := range x.Idx {
						g := grad * x.Val[k]
						// weight decay (L2) tocando apenas dimensões usadas
						m.W[c][j] = m.W[c][j]*(1.0-cfg.LR*cfg.L2) - cfg.LR*g
					}
				}
			}
		}

		// validação
		acc := evalAcc(m, Xv, Yv)
		// fmt.Printf("epoch %d  val_acc=%.4f\n", epoch+1, acc)
		if acc > bestValAcc {
			bestValAcc = acc
			best = *m
			noImprove = 0
		} else {
			noImprove++
			if noImprove >= cfg.Patience {
				break
			}
		}
	}
	*m = best
	return m, bestValAcc
}

func evalAcc(m *SoftmaxModel, X []SparseVec, Y []int) float64 {
	if len(X) == 0 {
		return 0
	}
	ok := 0
	for i := range X {
		lbl := m.predictOne(X[i])
		if lbl == m.Labels[Y[i]] {
			ok++
		}
	}
	return float64(ok) / float64(len(X))
}

// ===================== Modelo serializável =====================

type FullModel struct {
	Labels      []string      `json:"labels"`
	Vocab       Vocab         `json:"vocab"`
	SM          *SoftmaxModel `json:"softmax"`
	L2Normalize bool          `json:"l2_normalize"`
}

func saveModel(path string, fm *FullModel) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(fm)
}

func loadModel(path string) (*FullModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var fm FullModel
	if err := json.NewDecoder(f).Decode(&fm); err != nil {
		return nil, err
	}
	return &fm, nil
}

// ===================== Main =====================

func main() {
	predictText := flag.String("predict", "", "Texto para predição usando model.json")
	modelPath := flag.String("model", "model.json", "Caminho do modelo salvo/carregado")
	flag.Parse()

	if *predictText != "" {
		fm, err := loadModel(*modelPath)
		if err != nil {
			log.Fatalf("erro ao carregar modelo: %v", err)
		}
		x := vectorizeTFIDF_Sparse(fm.Vocab, *predictText, fm.L2Normalize)
		fmt.Printf("Texto: %q\nPredição => %s\n", *predictText, fm.SM.predictOne(x))
		return
	}

	// Treino
	samples, _ := loadJSONL("dataset.jsonl")
	train, test := trainTestSplit(samples, 0.2, 42)
	labels := uniqueLabels(samples)

	// Vocab com TF-IDF (minDF=1 para exemplo)
	var trainTexts []string
	for _, s := range train {
		trainTexts = append(trainTexts, s.Text)
	}
	vocab := buildVocabTFIDF(trainTexts, 1)
	l2normalize := true

	// Config mais rápida
	cfg := TrainCfg{
		LR:          0.2,
		L2:          1e-4,
		Epochs:      40,
		BatchSize:   256,
		Patience:    5,
		ValSplit:    0.15,
		Seed:        time.Now().Unix(),
		L2Normalize: l2normalize,
	}

	model, _ := trainSoftmax(cfg, labels, vocab, train)

	// Avaliação em teste
	Xte := make([]SparseVec, len(test))
	Yte := make([]string, len(test))
	for i, s := range test {
		Xte[i] = vectorizeTFIDF_Sparse(vocab, s.Text, l2normalize)
		Yte[i] = s.Label
	}
	pred := make([]string, len(test))
	for i := range Xte {
		pred[i] = model.predictOne(Xte[i])
	}
	acc := accuracy(Yte, pred)

	fmt.Println("Labels:", labels)
	fmt.Printf("Vocab size (1-gram+2-gram): %d\n", len(vocab.Words))
	fmt.Printf("Acurácia (teste): %.2f\n", acc)

	// Exemplos rápidos
	tests := []string{
		"meu limite diário do pix está baixo, aumenta para 800",
		"qual meu saldo disponível agora?",
		"envie 70 reais para a chave 11988887777",
	}
	for _, t := range tests {
		x := vectorizeTFIDF_Sparse(vocab, t, l2normalize)
		fmt.Printf("\nFrase: %q\n => %s\n", t, model.predictOne(x))
	}

	// Salvar
	fm := &FullModel{Labels: labels, Vocab: vocab, SM: model, L2Normalize: l2normalize}
	if err := saveModel("model.json", fm); err != nil {
		log.Fatalf("erro ao salvar modelo: %v", err)
	}
	fmt.Println("\nModelo salvo em model.json")
}

func loadJSONL(path string) ([]Sample, error) {
	b, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	// remove BOM se existir
	b = bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF})

	trimmed := strings.TrimSpace(string(b))
	if trimmed == "" {
		return nil, fmt.Errorf("arquivo vazio")
	}

	// 1) Se for um array JSON válido, parse direto
	if strings.HasPrefix(trimmed, "[") {
		var arr []Sample
		if err := json.Unmarshal([]byte(trimmed), &arr); err != nil {
			return nil, fmt.Errorf("JSON em array inválido: %w", err)
		}
		// filtra entradas vazias
		out := make([]Sample, 0, len(arr))
		for i, s := range arr {
			if strings.TrimSpace(s.Text) == "" || strings.TrimSpace(s.Label) == "" {
				continue
			}
			out = append(out, s)
			_ = i
		}
		return out, nil
	}

	return nil, fmt.Errorf("nenhum exemplo válido encontrado (arquivo pode conter apenas linhas vazias)")

}
