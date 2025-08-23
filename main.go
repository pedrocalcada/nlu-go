package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

type Sample struct {
	Text   string `json:"text"`
	Intent string `json:"intent"`
}

type NBModel struct {
	Classes            []string                  `json:"classes"`
	ClassCounts        map[string]int            `json:"class_counts"`
	TokenCounts        map[string]map[string]int `json:"token_counts"` // class -> token -> count
	Vocab              map[string]struct{}       `json:"vocab"`
	TotalTokensByClass map[string]int            `json:"total_tokens_by_class"`
	Laplace            float64                   `json:"laplace"`
	Stopwords          map[string]struct{}       `json:"stopwords"`
}

type ChatRequest struct {
	Message string `json:"message"`
}

func main() {
	// rand.Seed(time.Now().UnixNano())

	dataPath := "dataset.jsonl"
	samples, err := loadJSONL(dataPath)
	if err != nil {
		log.Fatalf("falha ao carregar dataset: %v", err)
	}

	// embaralha e split 80/20
	rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
	trainSize := int(float64(len(samples)) * 0.8)
	train := samples[:trainSize]
	test := samples[trainSize:]

	stop := buildStopwords()
	model := TrainNB(train, 1.0, stop) // Laplace=1.0

	acc, cm := Evaluate(model, test)
	fmt.Printf("Accuracy: %.4f\n", acc)
	fmt.Println("Matriz de confusão:")
	printConfusion(cm)

	// salva modelo
	if err := saveModel("model.json", model); err != nil {
		log.Fatalf("erro ao salvar modelo: %v", err)
	}
	fmt.Println("Modelo salvo em model.json")

	// exemplo de inferência
	ex := "envie 70 reais para a chave do carlos"
	intent, score := Predict(model, ex)
	fmt.Printf("Exemplo => \"%s\" -> intent=%s (score=%.4f)\n", ex, intent, score)

	// extração de entidades simples (ex.: valor, telefone, e-mail, CPF/CNPJ)
	ents := ExtractEntities(ex)
	fmt.Printf("Entidades: %+v\n", ents)

	http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		model, err := loadModel("model.json")
		if err != nil {
			panic(err)
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Failed to read body", http.StatusBadRequest)
			return
		}
		var req ChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		cls, p, margin, dist := PredictProba(model, req.Message)
		fmt.Printf("classe=%s prob=%.3f margin=%.2f dist=%v\n", cls, p, margin, dist)

		intent, score := Predict(model, req.Message)
		fmt.Printf("Texto: %q\n", req.Message)
		fmt.Printf("Intenção prevista: %s (score: %.4f)\n", intent, score)

		// 4) Extrair entidades
		ents := ExtractEntities(req.Message)
		fmt.Printf("Entidades: %+v\n", ents)
	})

	log.Println("Server running on :8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
}

func loadModel(path string) (*NBModel, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m NBModel
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// -------------------- IO --------------------

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
			if strings.TrimSpace(s.Text) == "" || strings.TrimSpace(s.Intent) == "" {
				continue
			}
			out = append(out, s)
			_ = i
		}
		return out, nil
	}

	// 2) Caso contrário, trata como JSONL (uma linha por objeto)
	lines := strings.Split(string(b), "\n")
	out := make([]Sample, 0, len(lines))
	for i, raw := range lines {
		lineno := i + 1
		line := strings.TrimSpace(strings.TrimRight(raw, "\r"))
		if line == "" {
			continue // pula linhas vazias
		}
		var s Sample
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			// Mostra um trecho da linha para facilitar o debug
			snippet := line
			if len(snippet) > 160 {
				snippet = snippet[:160] + "...(truncado)"
			}
			return nil, fmt.Errorf("linha %d inválida: %v\nconteúdo: %q", lineno, err, snippet)
		}
		// valida campos obrigatórios
		if strings.TrimSpace(s.Text) == "" || strings.TrimSpace(s.Intent) == "" {
			return nil, fmt.Errorf("linha %d: campos obrigatórios vazios (text/intent)", lineno)
		}
		out = append(out, s)
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("nenhum exemplo válido encontrado (arquivo pode conter apenas linhas vazias)")
	}
	return out, nil
}

func saveModel(path string, m *NBModel) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// -------------------- NB Treino/Inferência --------------------

func TrainNB(samples []Sample, laplace float64, stop map[string]struct{}) *NBModel {
	classCounts := map[string]int{}
	tokenCounts := map[string]map[string]int{}
	vocab := map[string]struct{}{}
	totalTokensByClass := map[string]int{}

	for _, s := range samples {
		classCounts[s.Intent]++
		if _, ok := tokenCounts[s.Intent]; !ok {
			tokenCounts[s.Intent] = map[string]int{}
		}
		toks := tokenize(s.Text, stop)
		for _, t := range toks {
			tokenCounts[s.Intent][t]++
			vocab[t] = struct{}{}
			totalTokensByClass[s.Intent]++
		}
	}

	classes := make([]string, 0, len(classCounts))
	for c := range classCounts {
		classes = append(classes, c)
	}
	sort.Strings(classes)

	return &NBModel{
		Classes:            classes,
		ClassCounts:        classCounts,
		TokenCounts:        tokenCounts,
		Vocab:              vocab,
		TotalTokensByClass: totalTokensByClass,
		Laplace:            laplace,
		Stopwords:          stop,
	}
}

func Predict(m *NBModel, text string) (string, float64) {
	toks := tokenize(text, m.Stopwords)

	// prior por classe
	totalDocs := 0
	for _, c := range m.ClassCounts {
		totalDocs += c
	}

	bestClass := ""
	bestScore := math.Inf(-1)

	V := float64(len(m.Vocab))

	for _, class := range m.Classes {
		logProb := math.Log(float64(m.ClassCounts[class]) / float64(totalDocs))
		den := float64(m.TotalTokensByClass[class]) + m.Laplace*V

		for _, t := range toks {
			tc := float64(m.TokenCounts[class][t]) + m.Laplace
			logProb += math.Log(tc / den)
		}

		if logProb > bestScore {
			bestScore = logProb
			bestClass = class
		}
	}
	return bestClass, bestScore
}

// retorna: classe vencedora, probabilidade dela, margem (logTop1 - logTop2), mapa classe->prob
func PredictProba(m *NBModel, text string) (string, float64, float64, map[string]float64) {
	toks := tokenize(text, m.Stopwords)

	totalDocs := 0
	for _, c := range m.ClassCounts {
		totalDocs += c
	}
	if totalDocs == 0 || len(m.Classes) == 0 {
		return "", 0, 0, nil
	}

	V := float64(len(m.Vocab))
	logScores := make([]float64, len(m.Classes))

	// calcula log-scores
	for i, class := range m.Classes {
		logProb := math.Log(float64(m.ClassCounts[class]) / float64(totalDocs))
		den := float64(m.TotalTokensByClass[class]) + m.Laplace*V
		for _, t := range toks {
			tc := float64(m.TokenCounts[class][t]) + m.Laplace
			logProb += math.Log(tc / den)
		}
		logScores[i] = logProb
	}

	// encontra top1/top2 para margem
	top1i, top2i := 0, 0
	for i := 1; i < len(logScores); i++ {
		if logScores[i] > logScores[top1i] {
			top2i = top1i
			top1i = i
		} else if top1i == top2i || logScores[i] > logScores[top2i] {
			top2i = i
		}
	}
	margin := logScores[top1i] - logScores[top2i]

	// normaliza com log-sum-exp -> probabilidades
	maxLog := logScores[top1i]
	var sumExp float64
	for _, v := range logScores {
		sumExp += math.Exp(v - maxLog)
	}
	probs := make(map[string]float64, len(m.Classes))
	for i, c := range m.Classes {
		probs[c] = math.Exp(logScores[i]-maxLog) / sumExp
	}

	winner := m.Classes[top1i]
	return winner, probs[winner], margin, probs
}

func Evaluate(m *NBModel, test []Sample) (float64, map[string]map[string]int) {
	if len(test) == 0 {
		return 0, nil
	}
	cm := map[string]map[string]int{}
	var correct int
	for _, s := range test {
		pred, _ := Predict(m, s.Text)
		if pred == s.Intent {
			correct++
		}
		if _, ok := cm[s.Intent]; !ok {
			cm[s.Intent] = map[string]int{}
		}
		cm[s.Intent][pred]++
	}
	return float64(correct) / float64(len(test)), cm
}

func printConfusion(cm map[string]map[string]int) {
	if cm == nil {
		fmt.Println("sem dados de avaliação")
		return
	}
	// coletar classes
	classSet := map[string]struct{}{}
	for g := range cm {
		classSet[g] = struct{}{}
		for p := range cm[g] {
			classSet[p] = struct{}{}
		}
	}
	var classes []string
	for c := range classSet {
		classes = append(classes, c)
	}
	sort.Strings(classes)

	// header
	fmt.Printf("%12s", "")
	for _, c := range classes {
		fmt.Printf("%12s", c)
	}
	fmt.Println()

	for _, g := range classes {
		fmt.Printf("%12s", g)
		for _, p := range classes {
			val := 0
			if row, ok := cm[g]; ok {
				val = row[p]
			}
			fmt.Printf("%12d", val)
		}
		fmt.Println()
	}
}

// -------------------- NLP util --------------------

var nonWord = regexp.MustCompile(`[^a-zA-Z0-9áàâãäéèêëíìîïóòôõöúùûüç]+`)

func tokenize(s string, stop map[string]struct{}) []string {
	s = strings.ToLower(s)
	s = nonWord.ReplaceAllString(s, " ")
	parts := strings.Fields(s)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if _, isStop := stop[p]; isStop {
			continue
		}
		// normalizações simples
		switch p {
		case "pix", "pix.", "pix,":
			p = "pix"
		}
		out = append(out, p)
	}
	return out
}

func buildStopwords() map[string]struct{} {
	words := []string{
		"o", "a", "os", "as", "de", "do", "da", "dos", "das", "um", "uma", "uns", "umas",
		"é", "ser", "está", "tá", "pra", "para", "por", "porfavor", "favor", "por-favor",
		"me", "eu", "vc", "você", "voce", "minha", "meu", "tem", "ter", "quero", "queria",
		"pode", "poder", "por", "em", "no", "na", "nos", "nas", "com", "sem", "daí", "ai",
		"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "please",
	}
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

// -------------------- Entidades por regras (exemplos úteis p/ Pix) --------------------

type Entities struct {
	AmountBRL string   `json:"amount_brl,omitempty"`
	Email     string   `json:"email,omitempty"`
	PhoneBR   string   `json:"phone_br,omitempty"`
	CPF       string   `json:"cpf,omitempty"`
	CNPJ      string   `json:"cnpj,omitempty"`
	PixKeys   []string `json:"pix_keys,omitempty"` // fallback com tudo que parece chave
}

var (
	reAmount = regexp.MustCompile(`\b(\d{1,3}(\.\d{3})*|\d+)(,\d{2})?\b`) // 70, 1.200, 15,50
	reEmail  = regexp.MustCompile(`[\w\.\-+]+@[\w\.\-]+\.[a-zA-Z]{2,}`)
	rePhone  = regexp.MustCompile(`\b(?:\+?55)?\s?(?:\(?\d{2}\)?\s?)?\d{4,5}[\s\-]?\d{4}\b`)
	reCPF    = regexp.MustCompile(`\b\d{3}\.?\d{3}\.?\d{3}\-?\d{2}\b`)
	reCNPJ   = regexp.MustCompile(`\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}\-?\d{2}\b`)
)

func ExtractEntities(s string) Entities {
	out := Entities{}
	if m := reAmount.FindString(s); m != "" {
		out.AmountBRL = m
	}
	if m := reEmail.FindString(s); m != "" {
		out.Email = m
	}
	if m := rePhone.FindString(s); m != "" {
		out.PhoneBR = m
	}
	if m := reCPF.FindString(s); m != "" {
		out.CPF = m
	}
	if m := reCNPJ.FindString(s); m != "" {
		out.CNPJ = m
	}
	// fallback: coleciona possíveis chaves (e-mail/telefone/cpf/cnpj já cobrem boa parte)
	keys := map[string]struct{}{}
	if out.Email != "" {
		keys[out.Email] = struct{}{}
	}
	if out.PhoneBR != "" {
		keys[out.PhoneBR] = struct{}{}
	}
	if out.CPF != "" {
		keys[out.CPF] = struct{}{}
	}
	if out.CNPJ != "" {
		keys[out.CNPJ] = struct{}{}
	}
	for k := range keys {
		out.PixKeys = append(out.PixKeys, k)
	}
	return out
}
