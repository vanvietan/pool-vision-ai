// Pool Vision AI backend: thin proxy in front of the Python AI service.
// Place where DB persistence + auth land later.
package main

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func aiServiceURL() string {
	if v := os.Getenv("AI_SERVICE_URL"); v != "" {
		return v
	}
	return "http://localhost:8000"
}

func main() {
	r := gin.Default()

	cfg := cors.DefaultConfig()
	cfg.AllowAllOrigins = true
	cfg.AllowMethods = []string{"GET", "POST", "OPTIONS"}
	cfg.AllowHeaders = []string{"*"}
	r.Use(cors.New(cfg))

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	r.POST("/analyze", analyzeHandler)

	addr := ":8080"
	if p := os.Getenv("PORT"); p != "" {
		addr = ":" + p
	}
	_ = r.Run(addr)
}

// analyzeHandler forwards the uploaded image to the AI service and streams the
// JSON response back to the client.
func analyzeHandler(c *gin.Context) {
	fileHeader, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing 'image' field"})
		return
	}

	src, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot read upload"})
		return
	}
	defer src.Close()

	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("image", fileHeader.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "build request failed"})
		return
	}
	if _, err := io.Copy(part, src); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "copy failed"})
		return
	}
	// forward optional manual table corners (JSON [[x,y],...]) if present
	if corners := c.PostForm("corners"); corners != "" {
		_ = w.WriteField("corners", corners)
	}
	w.Close()

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest(http.MethodPost, aiServiceURL()+"/analyze", &body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "request failed"})
		return
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "ai-service unreachable: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", data)
}
