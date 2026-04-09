require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.post('/api/analyze', async (req, res) => {
    try {
        const { query, marketContext } = req.body;
        
        if (!query) {
            return res.status(400).json({ error: "Query parameter is required." });
        }

        // Dynamically import ESM-only SDK
        const { SolRouter } = await import('@solrouter/sdk');
        
        const client = new SolRouter({
            apiKey: process.env.SOLROUTER_API_KEY
        });

        console.log(`[SolRouter Bridge] Received inference request context: ${marketContext || 'BTC 5M'}`);
        
        // Execute private query
        const response = await client.chat(query, {
            model: 'gpt-oss-20b',
            useLiveSearch: true
        });

        // The exact payload format depends on what client.chat returns (e.g. response.message vs response.choices)
        const content = response.message || (response.choices && response.choices[0] && response.choices[0].message.content) || response;

        res.json({
            success: true,
            sentimentSummary: content,
            attestation: response.privacyAttestationId
        });
        
    } catch (error) {
        console.error("[SolRouter Bridge] Error executing SolRouter request:", error.message);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`[SolRouter Bridge] Local Express Server securely running on http://localhost:${PORT}`);
});
