
import { GoogleGenAI, GenerateContentResponse, HarmCategory, HarmBlockThreshold, Content, Part } from '@google/genai';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactDOM from 'react-dom/client';

// --- Interface Definitions ---
interface FormData {
  age: string;
  currentWeight: string;
  targetWeight: string;
  height: string;
  activityLevel: 'sedentary' | 'light' | 'moderate' | 'active' | 'veryActive';
  goalTime: string;
  dislikedFoods: string;
  dislikedExercises: string;
  gender: 'male' | 'female' | 'other';
  desiredPhysique: string;
  dietaryRestrictions?: string; // Added for PlannerPage
}

interface Meal {
  name: string;
  description?: string;
  calories?: number;
}

interface DailyDiet {
  breakfast: Meal[];
  lunch: Meal[];
  dinner: Meal[];
  snacks?: Meal[];
}

interface Exercise {
  name: string;
  duration: string;
  setsReps?: string;
  notes?: string;
}

interface DetoxSuggestion {
  name: string;
  description: string;
  preparation?: string;
}

interface GeneratedPlan {
  planId: string;
  dietPlan: DailyDiet;
  exercisePlan: { day: string, activities: Exercise[] }[];
  detoxSuggestions: DetoxSuggestion[];
  motivationQuote: string;
  timeframeAssessment: string;
  estimatedTotalDailyCalories?: string; // Added for PlannerPage
}

interface Recipe {
  name:string;
  ingredients: string[];
  steps: string[];
  cookingTime?: string;
  servings?: string;
}

interface PriceAnalysisSource {
  web?: {
    uri: string;
    title: string;
  };
}
interface PriceAnalysisResult {
    text: string;
    sources: PriceAnalysisSource[];
}

interface BookSearchResult {
    id: string;
    title: string;
    author: string;
    description?: string;
    coverImageUrl?: string;
    freeSourceUrl?: string;
}

interface SavedBook extends BookSearchResult {
    generatedExcerpt?: string;
    totalPagesInExcerpt?: number;
    currentPageInExcerpt: number;
    lastReadTimestamp: number;
}

// YouTube Strategist Interfaces
interface NicheAnalysis {
    nicheSummary: string;
    popularSubTopics: string[];
    targetAudienceInsights: string;
    contentOpportunities: string[];
    keywords: string[];
}

interface HighlyViewedVideoExample {
    title: string;
    platform?: string; // e.g., YouTube, TikTok
    views?: string; // e.g., "1.2M views", "500K likes"
    link?: string;
    notes?: string; // Additional AI observations
}

interface PlatformDistribution {
    platformName: string; // e.g., YouTube, TikTok, Instagram Reels
    contentVolume: 'high' | 'medium' | 'low' | 'unknown';
    audienceEngagement: 'high' | 'medium' | 'low' | 'unknown';
    notes?: string;
}

interface MarketResearchData {
    analyzedNiche: string;
    highlyViewedVideos: HighlyViewedVideoExample[];
    platformAnalysis: PlatformDistribution[];
    generalObservations?: string;
    dataSourcesUsed?: string[]; // URLs or "Google Search"
}


interface BrollSuggestionLink {
    siteName: string; // e.g., "Pexels", "Pixabay"
    url: string;
}
interface BrollSuggestion {
    description: string; // e.g., "doğada yürüyen bir kişi"
    searchLinks: BrollSuggestionLink[];
}

interface StoryboardScene {
    sceneNumber: number;
    durationSeconds?: string;
    visualDescription: string;
    onScreenText?: string;
    voiceoverScript?: string;
    soundSuggestion?: string;
    brollSuggestions?: BrollSuggestion[];
}

interface ScriptSegment {
    segmentTitle: string;
    durationMinutes?: string;
    visualIdeas: string;
    voiceoverScript: string;
    brollSuggestions?: BrollSuggestion[];
}
interface ThumbnailConcept {
    conceptNumber: number;
    description: string;
    suggestedElements: string[] | string;
}

interface VideoBlueprint {
    generatedForNiche: string;
    videoType: 'reels' | 'long';
    videoTone?: string;
    titleSuggestions: string[];
    descriptionDraft: string;
    tagsKeywords: string[];
    storyboard?: StoryboardScene[];
    scriptSegments?: ScriptSegment[];
    fullVoiceoverScript?: string;
    fullSubtitleScript?: string;
    thumbnailConcepts: ThumbnailConcept[];
    soundtrackSuggestion?: string;
    potentialInteractionAssessment: string;
    aiToolSuggestions?: {
        thumbnailPrompts?: string[];
        voiceoverNotes?: string;
        visualPromptsForScenes?: { sceneNumber?: number; sceneDescription: string; promptSuggestion: string; }[];
    };
}

// --- End of Interface Definitions ---

// --- Constants ---
const API_MODEL = 'gemini-2.5-flash-preview-04-17';
const WORDS_PER_PAGE = 300;
const TARGET_LANGUAGES = [
    { code: "tr", name: "Türkçe" }, { code: "en", name: "English" }, { code: "de", name: "Deutsch (German)" },
    { code: "fr", name: "Français (French)" }, { code: "es", name: "Español (Spanish)" }, { code: "it", name: "Italiano (Italian)" },
    { code: "pt", name: "Português (Portuguese)" }, { code: "ru", name: "Русский (Russian)" }, { code: "ja", name: "日本語 (Japanese)" },
    { code: "ko", name: "한국어 (Korean)" }, { code: "zh", name: "中文 (Chinese)" }, { code: "ar", name: "العربية (Arabic)" },
];
const defaultSafetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];
// --- End of Constants ---

// --- API Client Initialization ---
const apiKeyFromEnv = process.env.API_KEY;
let genAI: GoogleGenAI | null = null;
let globalInitializationError: string | null = null;

if (!apiKeyFromEnv) {
  globalInitializationError = "API anahtarı bulunamadı. Lütfen .env dosyanızda API_KEY değişkenini ayarlayın.";
  console.error(globalInitializationError);
} else {
  try {
    genAI = new GoogleGenAI({ apiKey: apiKeyFromEnv });
  } catch (e: any) {
    console.error("GoogleGenAI istemcisi API anahtarı ile başlatılırken hata oluştu:", e);
    globalInitializationError = `API istemcisi başlatılamadı: ${e.message || String(e)}. Lütfen API anahtarınızı ve yapılandırmanızı kontrol edin.`;
  }
}
// --- End of API Client Initialization ---


// --- Utility Functions ---
const parseJsonResponse = <T,>(jsonString: string | undefined): T | null => {
    if (jsonString === undefined || jsonString === null || typeof jsonString !== 'string') {
        console.warn("parseJsonResponse: Boş veya tanımsız JSON dizesi alındı.");
        return null;
    }
    let str = jsonString.trim();
    const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
    const match = str.match(fenceRegex);
    if (match && match[2]) {
      str = match[2].trim(); 
    }

    if (str.startsWith("```json")) str = str.substring(7).trim();
    else if (str.startsWith("```")) str = str.substring(3).trim();
    if (str.endsWith("```")) str = str.substring(0, str.length - 3).trim();
    
    str = str.trim();
    
    if (!str.startsWith('{') && !str.startsWith('[')) {
        const firstBrace = str.indexOf('{');
        const firstBracket = str.indexOf('[');
        if (firstBrace === -1 && firstBracket === -1) {
            console.error("JSON parse error: Temizlenmiş dize '{' veya '[' içermiyor.", str.substring(0, 500));
            return null;
        }
        let startIndex = (firstBrace !== -1 && firstBracket !== -1) ? Math.min(firstBrace, firstBracket) : (firstBrace !== -1 ? firstBrace : firstBracket);
        if (startIndex === -1) { 
             console.error("JSON parse error: Unexpected state, start index not found.", str.substring(0, 500));
            return null;
        }
        str = str.substring(startIndex);
    }
    
    if (str.startsWith('{')) {
        const lastBrace = str.lastIndexOf('}');
        if (lastBrace !== -1) str = str.substring(0, lastBrace + 1);
    } else if (str.startsWith('[')) {
        const lastBracket = str.lastIndexOf(']');
        if (lastBracket !== -1) str = str.substring(0, lastBracket + 1);
    }
    str = str.trim();

    if (!str.startsWith('{') && !str.startsWith('[')) {
        console.error("JSON ayrıştırma hatası (sonraki deneme): Temizlenmiş dize '{' veya '[' ile başlamıyor.", str.substring(0, 500));
        return null;
    }
    try {
        return JSON.parse(str) as T;
    } catch (e: any) {
        console.error("JSON.parse son deneme hatası:", e.message, str.substring(0, 1000));
        return null;
    }
};

const copyToClipboard = useCallback((text: string) => {
    navigator.clipboard.writeText(text).then(() => {
        alert("Panoya kopyalandı!");
    }).catch(err => {
        console.error('Panoya kopyalama başarısız: ', err);
        alert("Panoya kopyalanamadı.");
    });
}, []);

// --- End of Utility Functions ---

// --- React Components ---

// --- Navigation Component ---
interface NavigationProps {
    currentPage: string;
    onNavigate: (page: 'planner' | 'priceAnalysis' | 'healthCheck' | 'bookReader' | 'youtubeStrategist') => void;
}

const Navigation: React.FC<NavigationProps> = ({ currentPage, onNavigate }) => {
    const pages = [
        { id: 'planner', label: 'Plan Oluşturucu' },
        { id: 'healthCheck', label: 'Sağlık Kontrolü' },
        { id: 'priceAnalysis', label: 'Fiyat Analizi' },
        { id: 'bookReader', label: 'Kitap Okuyucu' },
        { id: 'youtubeStrategist', label: 'AI Video Stratejisti' },
    ] as const;

    return (
        <nav className="main-nav">
            {pages.map(page => (
                <button
                    key={page.id}
                    className={currentPage === page.id ? 'nav-button active' : 'nav-button'}
                    onClick={() => onNavigate(page.id)}
                    aria-pressed={currentPage === page.id}
                    aria-label={`${page.label} sayfasına git`}
                >
                    {page.label}
                </button>
            ))}
        </nav>
    );
};
// --- End of Navigation Component ---


// --- PlannerPage Component ---
interface PlannerPageProps {
    genAI: GoogleGenAI | null;
    setError: (error: string | null) => void;
}
const PlannerPage: React.FC<PlannerPageProps> = ({ genAI, setError }) => {
    const [formData, setFormData] = useState<FormData>({
        age: '30', currentWeight: '70', targetWeight: '65', height: '170', activityLevel: 'moderate',
        goalTime: '3', dislikedFoods: 'Brokoli, Kereviz', dislikedExercises: 'Koşu bandı', gender: 'male',
        desiredPhysique: 'Atletik ve fit bir görünüm, karın kaslarımın belirginleşmesi',
        dietaryRestrictions: 'Yok',
    });
    const [generatedPlan, setGeneratedPlan] = useState<GeneratedPlan | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [recipeModalOpen, setRecipeModalOpen] = useState<boolean>(false);
    const [currentRecipe, setCurrentRecipe] = useState<Recipe | null>(null);
    const [recipeLoadingItemName, setRecipeLoadingItemName] = useState<string | null>(null);
    const [isGeneratingAlternative, setIsGeneratingAlternative] = useState<string | null>(null);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!genAI) { setError("API istemcisi başlatılamadı. API anahtarınızı kontrol edin."); return; }
        setIsLoading(true); setError(null); setGeneratedPlan(null);
        const prompt = `Bana aşağıdaki bilgilere sahip bir kişi için Türkçe olarak detaylı bir diyet ve fitness planı oluştur:
        Yaş: ${formData.age},
        Kilo: ${formData.currentWeight} kg,
        Hedef Kilo: ${formData.targetWeight} kg,
        Boy: ${formData.height} cm,
        Cinsiyet: ${formData.gender},
        Aktivite Seviyesi: ${formData.activityLevel},
        Hedef Süresi: ${formData.goalTime} ay,
        Sevilmeyen Yiyecekler: ${formData.dislikedFoods || 'Yok'},
        Ek Beslenme Tercihleri/Kısıtlamaları: ${formData.dietaryRestrictions || 'Yok'},
        Sevilmeyen Egzersizler: ${formData.dislikedExercises || 'Yok'},
        Hedeflenen Fizik ve Detaylar: ${formData.desiredPhysique}.

        Plan şunları içermelidir:
        1.  "planId": Rastgele bir UUID veya anlamlı bir string olabilir.
        2.  "dietPlan": Günlük öğünler (kahvaltı, öğle yemeği, akşam yemeği, ara öğünler) ve her öğün için yemek adı, açıklaması ve yaklaşık kalori miktarı.
        3.  "exercisePlan": Haftanın her günü için egzersiz aktiviteleri, süreleri, set/tekrar sayıları ve notlar.
        4.  "detoxSuggestions": 2-3 adet detoks veya sağlıklı içecek önerisi, açıklaması ve hazırlanışı.
        5.  "motivationQuote": Kişiyi motive edecek bir söz.
        6.  "timeframeAssessment": Hedefe ulaşma süresiyle ilgili kısa bir değerlendirme.
        7.  "estimatedTotalDailyCalories": Planın önerdiği yaklaşık toplam günlük kalori miktarı.
        JSON formatında yanıt ver (GeneratedPlan arayüzüne uygun).`;
        try {
            const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { responseMimeType: "application/json", safetySettings: defaultSafetySettings }});
            const planData = parseJsonResponse<GeneratedPlan>(response.text);
            if (planData) setGeneratedPlan(planData);
            else setError("AI'dan gelen plan yanıtı ayrıştırılamadı: " + response.text?.substring(0,200));
        } catch (err: any) { setError(`Plan oluşturma hatası: ${err.message || String(err)}`); }
        finally { setIsLoading(false); }
    };

    const handleGetRecipe = async (mealName: string, mealDescription?: string) => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        setRecipeLoadingItemName(mealName); setError(null); setCurrentRecipe(null); setRecipeModalOpen(true);
        const prompt = `"${mealName}" (${mealDescription || 'detay yok'}) için basit, sağlıklı bir yemek tarifi ver. JSON formatında yanıt ver (Recipe arayüzüne uygun). Tarif; yemek adı (name), malzemeler (ingredients: string[]), adımlar (steps: string[]), pişirme süresi (cookingTime: string) ve porsiyon (servings: string) bilgilerini içermelidir.`;
        try {
            const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { responseMimeType: "application/json", safetySettings: defaultSafetySettings }});
            const recipeData = parseJsonResponse<Recipe>(response.text);
            if (recipeData) setCurrentRecipe(recipeData);
            else { setError(`"${mealName}" için tarif yanıtı ayrıştırılamadı: ` + response.text?.substring(0,200)); setRecipeModalOpen(false); }
        } catch (err: any) { setError(`Tarif alma hatası: ${err.message}`); setRecipeModalOpen(false); }
        finally { setRecipeLoadingItemName(null); }
    };
    
    const handleGenerateAlternative = async (itemType: 'breakfast' | 'lunch' | 'dinner' | 'snack' | 'exercise', day?: string, originalName?: string, itemIndex?: number) => {
        if (!genAI || !generatedPlan) { setError("Plan veya API istemcisi mevcut değil."); return; }
        const loadingKey = itemType === 'exercise' ? `${itemType}_${day}_${itemIndex}` : `${itemType}_${itemIndex}`;
        setIsGeneratingAlternative(loadingKey); setError(null);
        let prompt = "";
        if (itemType === 'exercise' && day && originalName !== undefined && itemIndex !== undefined) {
          prompt = `Fitness planında "${day}" günü için "${originalName}" egzersizine alternatif bir egzersiz öner. JSON formatında (Exercise arayüzüne uygun - name, duration, setsReps, notes alanlarını içersin).`;
        } else if (originalName !== undefined && itemIndex !== undefined && ['breakfast', 'lunch', 'dinner', 'snack'].includes(itemType)) {
          const mealTypeTurkish = {'breakfast': 'kahvaltı', 'lunch': 'öğle yemeği', 'dinner': 'akşam yemeği', 'snack': 'ara öğün'}[itemType as 'breakfast' | 'lunch' | 'dinner' | 'snack'];
          prompt = `Diyet planındaki "${originalName}" (${mealTypeTurkish}) öğününe alternatif bir öğün öner. JSON formatında (Meal arayüzüne uygun - name, description, calories alanlarını içersin).`;
        } else { setError("Alternatif oluşturmak için yetersiz bilgi."); setIsGeneratingAlternative(null); return; }
        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { responseMimeType: "application/json", safetySettings: defaultSafetySettings }});
          if (itemType === 'exercise' && day && generatedPlan.exercisePlan && itemIndex !== undefined) {
            const altEx = parseJsonResponse<Exercise>(response.text);
            if (altEx && altEx.name && altEx.duration) { 
                const dayIdx = generatedPlan.exercisePlan.findIndex(d => d.day === day); 
                if (dayIdx!==-1) { 
                    const plan = {...generatedPlan}; 
                    plan.exercisePlan[dayIdx].activities[itemIndex] = altEx; 
                    setGeneratedPlan(plan);
                }
            } else setError("Alternatif egzersiz yanıtı ayrıştırılamadı veya eksik veri içeriyor: " + response.text?.substring(0,200));
          } else if (['breakfast', 'lunch', 'dinner', 'snack'].includes(itemType) && generatedPlan.dietPlan && itemIndex !== undefined) {
            const altMeal = parseJsonResponse<Meal>(response.text);
            if (altMeal && altMeal.name) { 
                const plan = {...generatedPlan}; 
                const cat = itemType as keyof DailyDiet; 
                if (plan.dietPlan[cat] && (plan.dietPlan[cat] as Meal[])[itemIndex]) {
                    (plan.dietPlan[cat] as Meal[])[itemIndex] = altMeal; 
                    setGeneratedPlan(plan);
                }
            } else setError("Alternatif öğün yanıtı ayrıştırılamadı veya eksik veri içeriyor: " + response.text?.substring(0,200));
          }
        } catch (err: any) { setError(`Alternatif oluşturma hatası: ${err.message}`); }
        finally { setIsGeneratingAlternative(null); }
    };

    return (
        <div className="page-content">
            <h2>Kişiye Özel Diyet ve Fitness Planı Oluşturucu</h2>
            <p>Sağlıklı bir yaşam için ilk adımınızı atın! Aşağıdaki formu doldurarak size özel diyet ve fitness planınızı oluşturabilirsiniz.</p>
            <form onSubmit={handleSubmit} className="form-container">
                <div className="form-group">
                    <label htmlFor="age">Yaşınız:</label>
                    <input type="number" id="age" name="age" value={formData.age} onChange={handleChange} required min="15" max="99" aria-label="Yaşınız"/>
                </div>
                <div className="form-group">
                    <label htmlFor="currentWeight">Mevcut Kilonuz (kg):</label>
                    <input type="number" id="currentWeight" name="currentWeight" value={formData.currentWeight} onChange={handleChange} required step="0.1" aria-label="Mevcut Kilonuz (kg)"/>
                </div>
                <div className="form-group">
                    <label htmlFor="targetWeight">Hedef Kilonuz (kg):</label>
                    <input type="number" id="targetWeight" name="targetWeight" value={formData.targetWeight} onChange={handleChange} required step="0.1" aria-label="Hedef Kilonuz (kg)"/>
                </div>
                <div className="form-group">
                    <label htmlFor="height">Boyunuz (cm):</label>
                    <input type="number" id="height" name="height" value={formData.height} onChange={handleChange} required aria-label="Boyunuz (cm)"/>
                </div>
                 <div className="form-group">
                    <label htmlFor="gender">Cinsiyetiniz:</label>
                    <select id="gender" name="gender" value={formData.gender} onChange={handleChange} aria-label="Cinsiyetiniz">
                        <option value="male">Erkek</option>
                        <option value="female">Kadın</option>
                        <option value="other">Diğer</option>
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="activityLevel">Aktivite Seviyeniz:</label>
                    <select id="activityLevel" name="activityLevel" value={formData.activityLevel} onChange={handleChange} aria-label="Aktivite Seviyeniz">
                        <option value="sedentary">Hareketsiz (Ofis işi vb.)</option>
                        <option value="light">Hafif Aktif (Haftada 1-2 gün egzersiz)</option>
                        <option value="moderate">Orta Derecede Aktif (Haftada 3-5 gün egzersiz)</option>
                        <option value="active">Aktif (Haftada 6-7 gün egzersiz)</option>
                        <option value="veryActive">Çok Aktif (Yoğun fiziksel iş veya günde iki antrenman)</option>
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="goalTime">Hedefinize Ulaşmak İstediğiniz Süre (Ay):</label>
                    <input type="number" id="goalTime" name="goalTime" value={formData.goalTime} onChange={handleChange} required min="1" aria-label="Hedefinize Ulaşmak İstediğiniz Süre (Ay)"/>
                </div>
                <div className="form-group large-input">
                    <label htmlFor="desiredPhysique">Ulaşmak İstediğiniz Fizik ve Detaylı Hedefleriniz:</label>
                    <textarea id="desiredPhysique" name="desiredPhysique" value={formData.desiredPhysique} onChange={handleChange} rows={3} placeholder="Örn: Daha sıkı bir vücut, karın kaslarının belirginleşmesi, dayanıklılığın artması..." aria-label="Ulaşmak İstediğiniz Fizik ve Detaylı Hedefleriniz"></textarea>
                </div>
                 <div className="form-group large-input">
                    <label htmlFor="dietaryRestrictions">Ek Beslenme Tercihleri/Kısıtlamaları (virgülle ayırın):</label>
                    <input type="text" id="dietaryRestrictions" name="dietaryRestrictions" value={formData.dietaryRestrictions} onChange={handleChange} placeholder="Örn: Vegan, glutensiz, laktozsuz" aria-label="Ek Beslenme Tercihleri/Kısıtlamaları"/>
                </div>
                <div className="form-group large-input">
                    <label htmlFor="dislikedFoods">Sevmediğiniz veya Alerjiniz Olan Yiyecekler (virgülle ayırın):</label>
                    <input type="text" id="dislikedFoods" name="dislikedFoods" value={formData.dislikedFoods} onChange={handleChange} placeholder="Örn: Pırasa, yer fıstığı" aria-label="Sevmediğiniz veya Alerjiniz Olan Yiyecekler"/>
                </div>
                <div className="form-group large-input">
                    <label htmlFor="dislikedExercises">Yapmaktan Hoşlanmadığınız Egzersizler (virgülle ayırın):</label>
                    <input type="text" id="dislikedExercises" name="dislikedExercises" value={formData.dislikedExercises} onChange={handleChange} placeholder="Örn: Mekik, uzun mesafe koşu" aria-label="Yapmaktan Hoşlanmadığınız Egzersizler"/>
                </div>
                <div className="form-group full-width-submit">
                    <button type="submit" className="submit-button" disabled={isLoading} aria-label="Planımı Oluştur">
                        {isLoading ? 'Plan Oluşturuluyor...' : 'Planımı Oluştur'}
                    </button>
                </div>
            </form>

            {isLoading && <div className="loading" role="status" aria-live="polite">Yapay zeka sizin için plan oluşturuyor, lütfen bekleyin...</div>}
           
            {generatedPlan && (
                <div className="generated-plan">
                    <div className="plan-id-section">
                        <h3>Plan Kimliği (ID)</h3>
                        <p className="plan-id-value">{generatedPlan.planId || "N/A"}</p>
                        <small>Bu kimliği not alarak planınız hakkında daha sonra sağlık analizi yapabilirsiniz.</small>
                    </div>

                    <div className="summary-section">
                        <h3><span role="img" aria-label="assessment">📝</span> Hedef Süre Değerlendirmesi</h3>
                        <p>{generatedPlan.timeframeAssessment || "AI bu konuda bir yorum yapmadı."}</p>
                    </div>
                     {generatedPlan.estimatedTotalDailyCalories && (
                        <div className="summary-section">
                            <h3><span role="img" aria-label="calories">🔥</span> Tahmini Günlük Kalori</h3>
                            <p>{generatedPlan.estimatedTotalDailyCalories}</p>
                        </div>
                    )}
                    
                    <div className="plan-section diet-plan">
                        <h2><span role="img" aria-label="apple">🍎</span> Diyet Planı</h2>
                        {(Object.keys(generatedPlan.dietPlan) as Array<keyof DailyDiet>).map(mealCategory => {
                             const meals = generatedPlan.dietPlan[mealCategory];
                             if (!meals || (Array.isArray(meals) && meals.length === 0)) return null;
                             const categoryName = mealCategory.charAt(0).toUpperCase() + mealCategory.slice(1);
                             let turkishCategoryName = categoryName;
                             if (categoryName === "Breakfast") turkishCategoryName = "Kahvaltı";
                             else if (categoryName === "Lunch") turkishCategoryName = "Öğle Yemeği";
                             else if (categoryName === "Dinner") turkishCategoryName = "Akşam Yemeği";
                             else if (categoryName === "Snacks") turkishCategoryName = "Ara Öğünler";

                            return (
                                <div key={mealCategory} className="meal-category">
                                    <h4>{turkishCategoryName}</h4>
                                    { (meals as Meal[]).map((meal, index) => (
                                        <div key={index} className="meal-item">
                                            <strong>{meal.name}</strong>
                                            {meal.description && <p>{meal.description}</p>}
                                            {meal.calories && <p><em>Kalori: ~{meal.calories}</em></p>}
                                            <button
                                                onClick={() => handleGetRecipe(meal.name, meal.description)}
                                                className="action-button"
                                                disabled={recipeLoadingItemName === meal.name}
                                                aria-label={`${meal.name} için tarif al`} >
                                                {recipeLoadingItemName === meal.name ? 'Tarif Yükleniyor...' : 'Tarif Al'}
                                            </button>
                                            <button
                                                onClick={() => handleGenerateAlternative(mealCategory as any, undefined, meal.name, index)}
                                                className="action-button alternative"
                                                disabled={isGeneratingAlternative === `${mealCategory}_${index}`}
                                                aria-label={`${meal.name} için alternatif öğün oluştur`} >
                                                {isGeneratingAlternative === `${mealCategory}_${index}` ? 'Alternatif Aranıyor...' : 'Alternatif Bul'}
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            );
                        })}
                    </div>

                    <div className="plan-section exercise-plan">
                        <h2><span role="img" aria-label="muscle">💪</span> Egzersiz Planı</h2>
                        {generatedPlan.exercisePlan.map((dayPlan, dayIndex) => (
                            <div key={dayIndex} className="plan-item">
                                <h4>{dayPlan.day}</h4>
                                {dayPlan.activities.map((activity, activityIndex) => (
                                    <div key={activityIndex} className="exercise-item">
                                        <strong>{activity.name}</strong>
                                        <p>Süre: {activity.duration}</p>
                                        {activity.setsReps && <p>Set/Tekrar: {activity.setsReps}</p>}
                                        {activity.notes && <p><em>Notlar: {activity.notes}</em></p>}
                                        <button
                                            onClick={() => handleGenerateAlternative('exercise', dayPlan.day, activity.name, activityIndex)}
                                            className="action-button alternative"
                                            disabled={isGeneratingAlternative === `exercise_${dayPlan.day}_${activityIndex}`}
                                            aria-label={`${activity.name} için alternatif egzersiz oluştur`} >
                                            {isGeneratingAlternative === `exercise_${dayPlan.day}_${activityIndex}` ? 'Alternatif Aranıyor...' : 'Alternatif Egzersiz Bul'}
                                        </button>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>

                     {generatedPlan.detoxSuggestions && generatedPlan.detoxSuggestions.length > 0 && (
                        <div className="plan-section detox-suggestions">
                            <h2><span role="img" aria-label="lemon">🍋</span> Detoks ve İçecek Önerileri</h2>
                            {generatedPlan.detoxSuggestions.map((suggestion, index) => (
                                <div key={index} className="plan-item">
                                    <strong>{suggestion.name}</strong>
                                    <p>{suggestion.description}</p>
                                    {suggestion.preparation && <p><em>Hazırlanışı: {suggestion.preparation}</em></p>}
                                </div>
                            ))}
                        </div>
                    )}
                     <div className="motivation-section">
                        <h3><span role="img" aria-label="star">🌟</span> Motivasyon Sözü</h3>
                        <p><em>"{generatedPlan.motivationQuote || "Harika bir iş çıkarıyorsun!"}"</em></p>
                    </div>
                </div>
            )}

            {recipeModalOpen && (
                <div className="modal" onClick={() => setRecipeModalOpen(false)} role="dialog" aria-modal="true" aria-labelledby="recipeModalTitle">
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <button className="modal-close" onClick={() => setRecipeModalOpen(false)} aria-label="Tarif penceresini kapat">&times;</button>
                        {currentRecipe ? (
                            <>
                                <h3 id="recipeModalTitle">{currentRecipe.name} Tarifi</h3>
                                {currentRecipe.cookingTime && <p><strong>Hazırlık/Pişirme Süresi:</strong> {currentRecipe.cookingTime}</p>}
                                {currentRecipe.servings && <p><strong>Porsiyon:</strong> {currentRecipe.servings}</p>}
                                <h4>Malzemeler:</h4>
                                <ul>
                                    {currentRecipe.ingredients.map((ing, i) => <li key={i}>{ing}</li>)}
                                </ul>
                                <h4>Adımlar:</h4>
                                <ol>
                                    {currentRecipe.steps.map((step, i) => <li key={i}>{step}</li>)}
                                </ol>
                            </>
                        ) : (
                            <div className="loading" role="status" aria-live="polite">Tarif yükleniyor...</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
// --- End of PlannerPage Component ---


// --- HealthCheckPage Component ---
interface HealthCheckPageProps {
    genAI: GoogleGenAI | null;
    setError: (error: string | null) => void;
}
const HealthCheckPage: React.FC<HealthCheckPageProps> = ({ genAI, setError }) => {
    const [dietPlanIdInput, setDietPlanIdInput] = useState<string>('');
    const [healthConditionsInput, setHealthConditionsInput] = useState<string>('');
    const [healthAnalysisResult, setHealthAnalysisResult] = useState<string | null>(null);
    const [isAnalyzingHealth, setIsAnalyzingHealth] = useState<boolean>(false);

    const handleHealthAnalysis = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!dietPlanIdInput.trim() && !healthConditionsInput.trim()) { setError("Analiz için Plan ID'si veya sağlık durumları girin."); return; }
        setIsAnalyzingHealth(true); setError(null); setHealthAnalysisResult(null);
        
        let prompt = `LÜTFEN DİKKAT: Bu analiz tıbbi tavsiye niteliği taşımaz ve sadece genel bilgilendirme ve farkındalık yaratma amaçlıdır. Herhangi bir sağlık sorununuz varsa veya yeni bir diyet/egzersiz programına başlamadan önce MUTLAKA bir doktora veya diyetisyene danışın. Yanıtınızı Türkçe olarak, Markdown formatında (başlıklar, alt başlıklar, listeler kullanarak) ve anlaşılır bir şekilde sunun.\n\nBir sağlık danışmanı gibi davranarak aşağıdaki bilgileri analiz et:\n`;

        if (dietPlanIdInput.trim()) {
            prompt += `\n## Diyet Planı ID: ${dietPlanIdInput} için Genel Değerlendirme\nBu ID'ye sahip bir diyet planının genel olarak potansiyel etkilerini değerlendir. Bu ID'nin spesifik içeriğini bilmediğini, bu yüzden sadece genel varsayımlar (örneğin, dengeli bir plan olduğu veya belirli bir amaca yönelik olabileceği) üzerinden yorum yapacağını belirt. Olası olumlu ve olumsuz yönleri (enerji seviyesi, besin çeşitliliği, sürdürülebilirlik gibi genel başlıklar altında) ele al.`;
        }
        if (healthConditionsInput.trim()) {
            prompt += `\n\n## Mevcut Sağlık Durumları: "${healthConditionsInput}" için Genel Öneriler\nBu sağlık durumlarına sahip bir birey için genel beslenme ve yaşam tarzı önerilerinde bulun. Özellikle dikkat edilmesi gereken noktaları, kaçınılması veya tercih edilmesi gereken besin gruplarını (genel olarak) vurgula.`;
        }
        if (dietPlanIdInput.trim() && healthConditionsInput.trim()){
             prompt += `\n\n## Plan ve Sağlık Durumu Sentezi\nYukarıdaki diyet planı ID'si hakkındaki genel varsayımlarını ve belirtilen sağlık durumlarını birleştirerek, bu kişinin genel olarak nelere dikkat etmesi gerektiği konusunda bir sentez yap. Potansiyel riskleri ve faydaları dengeli bir şekilde ele al.`;
        }
        prompt += `\n\n---\n**ÖNEMLİ UYARI:** Bu analiz, yalnızca yapay zeka tarafından üretilmiş genel bilgiler içerir ve kişiye özel tıbbi tavsiye yerine geçmez. Sağlığınızla ilgili herhangi bir karar almadan önce mutlaka yetkili bir sağlık profesyoneline danışınız.`;

        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: {safetySettings: defaultSafetySettings} });
          setHealthAnalysisResult(response.text);
        } catch (err: any) { setError(`Sağlık analizi hatası: ${err.message}`); }
        finally { setIsAnalyzingHealth(false); }
      };

    return (
        <div className="page-content">
            <h2>Diyet Planı ve Sağlık Durumu Analizi</h2>
            <p>Mevcut diyet planınızın ID'sini ve/veya bilinen sağlık durumlarınızı girerek yapay zekadan genel bir analiz ve öneri alabilirsiniz. <strong>Bu analiz tıbbi tavsiye yerine geçmez.</strong></p>
            <div className="form-container single-column-form">
                <div className="form-group">
                    <label htmlFor="dietPlanIdInput">Diyet Planı ID'si (isteğe bağlı):</label>
                    <input
                        type="text"
                        id="dietPlanIdInput"
                        value={dietPlanIdInput}
                        onChange={(e) => setDietPlanIdInput(e.target.value)}
                        placeholder="Oluşturduğunuz planın ID'sini girin"
                        aria-label="Diyet Planı ID'si"
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="healthConditionsInput">Bilinen Sağlık Durumlarınız (isteğe bağlı):</label>
                    <textarea
                        id="healthConditionsInput"
                        value={healthConditionsInput}
                        onChange={(e) => setHealthConditionsInput(e.target.value)}
                        rows={4}
                        placeholder="Örn: Diyabet, yüksek tansiyon, laktoz intoleransı..."
                        aria-label="Bilinen Sağlık Durumlarınız"
                    />
                </div>
                 <button onClick={handleHealthAnalysis} className="submit-button" disabled={isAnalyzingHealth || (!dietPlanIdInput.trim() && !healthConditionsInput.trim())} aria-label="Sağlık Durumumu ve Planımı Analiz Et">
                    {isAnalyzingHealth ? 'Analiz Ediliyor...' : 'Sağlık Durumumu ve Planımı Analiz Et'}
                </button>
            </div>

            {isAnalyzingHealth && <div className="loading" role="status" aria-live="polite">Sağlık verileriniz analiz ediliyor...</div>}
            
            {healthAnalysisResult && (
                <div className="health-analysis-results page-section">
                    <h3><span role="img" aria-label="stethoscope">🩺</span> AI Sağlık Analizi Sonucu:</h3>
                    <div className="ai-generated-text">
                        {healthAnalysisResult}
                    </div>
                    <p className="warning-text"><strong>Uyarı:</strong> Bu analiz sadece bilgilendirme amaçlıdır ve bir doktor tavsiyesi değildir. Sağlığınızla ilgili kararlar almadan önce mutlaka bir sağlık profesyoneline danışınız.</p>
                </div>
            )}
        </div>
    );
};
// --- End of HealthCheckPage Component ---


// --- PriceAnalysisPage Component ---
interface PriceAnalysisPageProps {
    genAI: GoogleGenAI | null;
    setError: (error: string | null) => void;
}
const PriceAnalysisPage: React.FC<PriceAnalysisPageProps> = ({ genAI, setError }) => {
    const [productQuery, setProductQuery] = useState<string>('RTX 4090 ekran kartı fiyatları');
    const [analysisRegion, setAnalysisRegion] = useState<string>('Türkiye');
    const [priceAnalysisResults, setPriceAnalysisResults] = useState<PriceAnalysisResult | null>(null);
    const [isAnalyzingPrice, setIsAnalyzingPrice] = useState<boolean>(false);

    const handlePriceAnalysis = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!productQuery.trim()) { setError("Analiz edilecek ürün/hizmeti girin."); return; }
        setIsAnalyzingPrice(true); setError(null); setPriceAnalysisResults(null);
        const prompt = `"${productQuery}" için "${analysisRegion}" bölgesinde/ülkesinde detaylı bir fiyat analizi yap. Analizin şunları içermeli:
        1.  Genel piyasa fiyat aralığı (varsa farklı modeller/seviyeler için).
        2.  Mümkünse en iyi fırsatları bulabileceğin yerler veya platform türleri hakkında genel bilgi.
        3.  Bu ürünün/hizmetin fiyatını etkileyen başlıca faktörler (arz-talep, marka, özellikler, sezonluk durumlar, bölgesel vergiler/gümrük vb.).
        4.  Google Search aracını kullanarak bu ürün/hizmetle ilgili "${analysisRegion}" özelindeki güncel haberleri, trendleri veya önemli piyasa gelişmelerini (son 1-3 ay içindeki) özetle.
        Cevabını Türkçe ve anlaşılır bir dille yaz. Sonuçları bir analiz metni ve eğer Google Search'ten bilgi bulunduysa, kullanılan kaynakların bir listesi (başlık ve URI içerecek şekilde) olarak sun. Kaynak listesi boş olabilir.`;
        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { tools: [{ googleSearch: {} }], safetySettings: defaultSafetySettings }});
          const sources: PriceAnalysisSource[] = response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map((chunk: any) => ({web: {uri: chunk.web?.uri || '', title: chunk.web?.title || 'Başlık Yok'}})) || [];
          setPriceAnalysisResults({ text: response.text, sources });
        } catch (err: any) { setError(`Fiyat analizi hatası: ${err.message}`); }
        finally { setIsAnalyzingPrice(false); }
      };

    return (
        <div className="page-content">
            <h2>Ürün/Hizmet Fiyat Analizi (Google Search Destekli)</h2>
            <p>Merak ettiğiniz bir ürünün veya hizmetin piyasa fiyatını, güncel trendlerini ve fiyatını etkileyen faktörleri AI ve Google Search entegrasyonu ile analiz edin.</p>
            <div className="form-container single-column-form">
                <div className="form-group">
                    <label htmlFor="productQuery">Analiz Edilecek Ürün/Hizmet Adı:</label>
                    <input
                        type="text"
                        id="productQuery"
                        value={productQuery}
                        onChange={(e) => setProductQuery(e.target.value)}
                        placeholder="Örn: En son model akıllı telefon, İstanbul-Ankara uçak bileti"
                        aria-label="Analiz Edilecek Ürün veya Hizmet Adı"
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="analysisRegion">Analiz Bölgesi/Ülkesi:</label>
                    <input
                        type="text"
                        id="analysisRegion"
                        value={analysisRegion}
                        onChange={(e) => setAnalysisRegion(e.target.value)}
                        placeholder="Örn: Türkiye, Almanya, İstanbul"
                        aria-label="Analiz Bölgesi veya Ülkesi"
                    />
                </div>
                <button onClick={handlePriceAnalysis} className="submit-button" disabled={isAnalyzingPrice || !productQuery.trim()} aria-label="Fiyat Analizi Yap">
                    {isAnalyzingPrice ? 'Fiyatlar Analiz Ediliyor...' : 'Fiyat Analizi Yap'}
                </button>
            </div>

            {isAnalyzingPrice && <div className="loading" role="status" aria-live="polite">Ürün/hizmet için fiyat analizi yapılıyor...</div>}

            {priceAnalysisResults && (
                <div className="price-analysis-results page-section">
                    <h3><span role="img" aria-label="chart">📊</span> Fiyat Analizi Sonuçları ({analysisRegion}):</h3>
                    <div className="ai-text-output">
                        <h4>AI Analizi:</h4>
                        <div className="ai-generated-text">{priceAnalysisResults.text}</div>
                    </div>
                    {priceAnalysisResults.sources && priceAnalysisResults.sources.length > 0 && (
                        <div className="sources-section">
                            <h4><span role="img" aria-label="link">🔗</span> Kullanılan Kaynaklar (Google Search):</h4>
                            <ul>
                                {priceAnalysisResults.sources.filter(source => source.web && source.web.uri).map((source, index) => (
                                    <li key={index}>
                                        <a href={source.web!.uri} target="_blank" rel="noopener noreferrer">
                                            {source.web!.title || source.web!.uri}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
// --- End of PriceAnalysisPage Component ---


// --- BookReaderPage Component ---
interface BookReaderPageProps {
    genAI: GoogleGenAI | null;
    setError: (error: string | null) => void;
}
const BookReaderPage: React.FC<BookReaderPageProps> = ({ genAI, setError }) => {
    const [bookSearchQuery, setBookSearchQuery] = useState<string>('Sherlock Holmes');
    const [bookSearchResults, setBookSearchResults] = useState<BookSearchResult[]>([]);
    const [savedBooks, setSavedBooks] = useState<SavedBook[]>(() => {
        const loaded = localStorage.getItem('savedBooks');
        if (loaded) {
            try { return (JSON.parse(loaded) as SavedBook[]).map(b => ({...b, currentPageInExcerpt: b.currentPageInExcerpt || 0})); }
            catch (e) { console.error("Kaydedilmiş kitaplar yüklenirken hata:", e); localStorage.removeItem('savedBooks'); return []; }
        } return [];
    });
    const [currentReadingBook, setCurrentReadingBook] = useState<SavedBook | null>(null);
    const [readerContent, setReaderContent] = useState<string[]>([]); // Array of strings, each string is a page
    const [isFetchingBookContent, setIsFetchingBookContent] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(false); // For book search loading
    const [selectedTargetLanguage, setSelectedTargetLanguage] = useState<string>("tr");
    const [isTranslatingBook, setIsTranslatingBook] = useState(false);

    useEffect(() => { localStorage.setItem('savedBooks', JSON.stringify(savedBooks)); }, [savedBooks]);

    const handleBookSearch = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!bookSearchQuery.trim()) { setError("Aranacak kitap adı/yazarını girin."); return; }
        setIsLoading(true); setError(null); setBookSearchResults([]); setCurrentReadingBook(null);
        const prompt = `"${bookSearchQuery}" ile ilgili kitapları bul. JSON formatında bir dizi olarak yanıt ver. Her kitap için şu bilgileri içermeli: id (benzersiz bir string, örn: ISBN veya rastgele bir uuid), title, author, description (kısa bir özet), coverImageUrl (varsa bir kapak resmi URL'si, yoksa boş string), freeSourceUrl (varsa kitabın ücretsiz ve yasal olarak okunabileceği bir URL, yoksa boş string). En az 3, en fazla 7 sonuç döndür. Google Search aracını kullanarak sonuçları zenginleştir. Türkçe yanıtla.`;
        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { tools: [{ googleSearch: {} }], safetySettings: defaultSafetySettings }});
          const books = parseJsonResponse<BookSearchResult[]>(response.text);
          if (books && Array.isArray(books)) setBookSearchResults(books);
          else setError("Kitap arama sonuçları hatalı veya boş döndü: " + response.text?.substring(0,200));
        } catch (err: any) { setError(`Kitap arama hatası: ${err.message}`); }
        finally { setIsLoading(false); }
    };

    const saveBookToLibrary = (book: BookSearchResult) => {
        const newSavedBook: SavedBook = { ...book, currentPageInExcerpt: 0, lastReadTimestamp: Date.now() };
        setSavedBooks(prev => {
            if (prev.find(b => b.id === newSavedBook.id)) return prev; 
            const updatedBooks = [...prev, newSavedBook];
            updatedBooks.sort((a,b) => (b.lastReadTimestamp || 0) - (a.lastReadTimestamp || 0));
            return updatedBooks;
        });
    };

    const openBookReader = async (book: SavedBook) => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        setCurrentReadingBook(book); setReaderContent([]); setIsFetchingBookContent(true); setError(null);
        
        setSavedBooks(prev => prev.map(sb => sb.id === book.id ? { ...sb, lastReadTimestamp: Date.now() } : sb).sort((a,b) => (b.lastReadTimestamp || 0) - (a.lastReadTimestamp || 0)));

        if (book.generatedExcerpt && book.totalPagesInExcerpt) {
          const pages = []; for (let i=0; i<book.totalPagesInExcerpt; i++) pages.push(book.generatedExcerpt.substring(i*WORDS_PER_PAGE, (i+1)*WORDS_PER_PAGE));
          setReaderContent(pages); setIsFetchingBookContent(false); return;
        }
        const prompt = `"${book.title}" (${book.author}) adlı kitaptan, başlangıcından itibaren anlamlı bir bölüm olacak şekilde yaklaşık ${WORDS_PER_PAGE * 5} kelimelik bir metin ver. Sadece kitabın metnini, herhangi bir ek açıklama veya başlık olmadan düz metin olarak Türkçe ver.`;
        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { safetySettings: defaultSafetySettings } });
          const excerpt = response.text.trim(); 
          if (!excerpt) {
            setError("Kitap için anlamlı bir bölüm alınamadı.");
            setReaderContent(["Bu kitap için okunacak içerik bulunamadı veya AI tarafından üretilemedi."]);
            setIsFetchingBookContent(false);
            return;
          }
          const totalPages = Math.ceil(excerpt.length / WORDS_PER_PAGE);
          const pages = []; for (let i=0; i<totalPages; i++) pages.push(excerpt.substring(i*WORDS_PER_PAGE, (i+1)*WORDS_PER_PAGE));
          setReaderContent(pages.length > 0 ? pages : ["Bu kitap için okunacak içerik bulunamadı."]);
          setSavedBooks(prev => prev.map(sb => sb.id === book.id ? { ...sb, generatedExcerpt: excerpt, totalPagesInExcerpt: totalPages } : sb));
        } catch (err: any) { setError(`Kitap içeriği alma hatası: ${err.message}`); setReaderContent(["İçerik yüklenirken bir hata oluştu."]); }
        finally { setIsFetchingBookContent(false); }
    };

    const handlePageChange = (bookId: string, newPage: number) => {
        setSavedBooks(prev => prev.map(book => book.id === bookId ? { ...book, currentPageInExcerpt: newPage, lastReadTimestamp: Date.now() } : book ));
        setCurrentReadingBook(prev => prev && prev.id === bookId ? { ...prev, currentPageInExcerpt: newPage } : prev);
    };
    
    const handleTranslateBookPage = async () => {
        if (!genAI || !currentReadingBook || readerContent.length === 0 || !selectedTargetLanguage) { setError("Çeviri için bilgi eksik."); return; }
        if (currentReadingBook.currentPageInExcerpt < 0 || currentReadingBook.currentPageInExcerpt >= readerContent.length) { setError("Geçersiz sayfa numarası."); return; }
        setIsTranslatingBook(true); setError(null);
        const currentPageText = readerContent[currentReadingBook.currentPageInExcerpt];
        const targetLanguageName = TARGET_LANGUAGES.find(lang => lang.code === selectedTargetLanguage)?.name || selectedTargetLanguage;
        if (currentPageText.includes(`--- (${targetLanguageName} diline çevrildi) ---`)){
            setIsTranslatingBook(false); 
            return;
        }
        const prompt = `Aşağıdaki metni ${targetLanguageName} diline çevir. Sadece çevrilmiş metni döndür:\n\n${currentPageText.split(`\n\n--- (`)[0]}`; // Translate only original text
        try {
          const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: {safetySettings: defaultSafetySettings} });
          const translatedText = response.text.trim();
          const newReaderContent = [...readerContent];
          newReaderContent[currentReadingBook.currentPageInExcerpt] = translatedText + `\n\n--- (${targetLanguageName} diline çevrildi) ---`;
          setReaderContent(newReaderContent);
        } catch (err: any) { setError(`Sayfa çevirme hatası: ${err.message}`); }
        finally { setIsTranslatingBook(false); }
    };
    
    const removeFromLibrary = (bookId: string) => {
        if (window.confirm("Bu kitabı kütüphaneden silmek istediğinize emin misiniz?")) {
            setSavedBooks(prev => prev.filter(b => b.id !== bookId));
            if (currentReadingBook && currentReadingBook.id === bookId) {
                setCurrentReadingBook(null);
                setReaderContent([]);
            }
        }
    };

    return (
        <div className="page-content">
            <h2>Kitap Okuyucu ve Kütüphane (Google Search Destekli)</h2>
            {!currentReadingBook ? (
                <>
                    <p>İlgilendiğiniz kitapları arayın, kütüphanenize ekleyin ve AI tarafından oluşturulan bölümlerini okuyun.</p>
                    <div className="form-container single-column-form">
                        <div className="form-group">
                            <label htmlFor="bookSearchQuery">Kitap Adı veya Yazar:</label>
                            <input
                                type="text"
                                id="bookSearchQuery"
                                value={bookSearchQuery}
                                onChange={(e) => setBookSearchQuery(e.target.value)}
                                placeholder="Örn: Yüzüklerin Efendisi, Tolstoy"
                                aria-label="Aranacak Kitap Adı veya Yazar"
                            />
                        </div>
                        <button onClick={handleBookSearch} className="submit-button" disabled={isLoading || !bookSearchQuery.trim()} aria-label="Kitap Ara">
                            {isLoading ? 'Kitaplar Aranıyor...' : 'Kitap Ara'}
                        </button>
                    </div>

                    {isLoading && <div className="loading" role="status" aria-live="polite">Kitaplar aranıyor...</div>}

                    {bookSearchResults.length > 0 && (
                        <div className="page-section">
                            <h3>Arama Sonuçları:</h3>
                            <div className="item-list media-item-list">
                                {bookSearchResults.map(book => (
                                    <div key={book.id} className="media-item book-item">
                                        {book.coverImageUrl ? <img src={book.coverImageUrl} alt={`${book.title} kapak`} className="item-cover-image" onError={(e) => (e.currentTarget.style.display = 'none')} /> : <div className="item-cover-image-placeholder">Kapak Yok</div>}
                                        <h4>{book.title}</h4>
                                        <p className="item-author-year">{book.author}</p>
                                        {book.description && <p className="item-description-small" title={book.description}>{book.description}</p>}
                                        <div className="item-actions">
                                            <button onClick={() => saveBookToLibrary(book)} className="action-button" disabled={savedBooks.some(sb => sb.id === book.id)} aria-label={`${book.title} kitabını kütüphaneye ekle`}>
                                                {savedBooks.some(sb => sb.id === book.id) ? 'Kütüphanede' : 'Kütüphaneye Ekle'}
                                            </button>
                                            {book.freeSourceUrl && 
                                                <a href={book.freeSourceUrl} target="_blank" rel="noopener noreferrer" className="action-button free-source-button" aria-label={`${book.title} için ücretsiz kaynağa git`}>
                                                  Ücretsiz Kaynağa Git <span aria-hidden="true">↗</span>
                                                </a>
                                            }
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {savedBooks.length > 0 && (
                        <div className="page-section">
                            <h3>Kütüphanem (Son Okunanlar Üstte):</h3>
                            <div className="item-list media-item-list">
                                {savedBooks.map(book => (
                                    <div key={book.id} className="media-item book-item">
                                       {book.coverImageUrl ? <img src={book.coverImageUrl} alt={`${book.title} kapak`} className="item-cover-image" onError={(e) => (e.currentTarget.style.display = 'none')} /> : <div className="item-cover-image-placeholder">Kapak Yok</div>}
                                        <h4>{book.title}</h4>
                                        <p className="item-author-year">{book.author}</p>
                                        <div className="item-actions">
                                            <button onClick={() => openBookReader(book)} className="action-button" aria-label={`${book.title} kitabını oku`}>Oku</button>
                                            <button onClick={() => removeFromLibrary(book.id)} className="action-button alternative remove-button" aria-label={`${book.title} kitabını kütüphaneden sil`}>Kütüphaneden Sil</button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                     {bookSearchResults.length === 0 && savedBooks.length === 0 && !isLoading && (
                        <p style={{textAlign:'center', marginTop:'20px'}}>Arama yapın veya daha önce eklediğiniz kitapları görüntüleyin.</p>
                    )}
                </>
            ) : (
                <div className="media-reader-view">
                    <div className="reader-view-header">
                        <h3>{currentReadingBook.title}</h3>
                        <p className="reader-author">Yazar: {currentReadingBook.author}</p>
                    </div>
                     <button onClick={() => {setCurrentReadingBook(null); setReaderContent([]);}} className="action-button back-to-library-button" aria-label="Kütüphaneye geri dön">
                        <span aria-hidden="true">&larr;</span> Kütüphaneye Dön
                     </button>
                     
                     <div className="language-actions">
                        <span>Sayfayı Çevir:</span>
                        <select value={selectedTargetLanguage} onChange={(e) => setSelectedTargetLanguage(e.target.value)} aria-label="Hedef çeviri dili">
                            {TARGET_LANGUAGES.map(lang => (
                                <option key={lang.code} value={lang.code}>{lang.name}</option>
                            ))}
                        </select>
                        <button onClick={handleTranslateBookPage} disabled={isTranslatingBook} className="action-button" aria-label="Seçili dile çevir">
                            {isTranslatingBook ? 'Çevriliyor...' : 'Çevir'}
                        </button>
                    </div>

                    {isFetchingBookContent ? (
                        <div className="loading" role="status" aria-live="polite">Kitap içeriği yükleniyor...</div>
                    ) : readerContent.length > 0 && readerContent[0] !== "Bu kitap için okunacak içerik bulunamadı." && readerContent[0] !== "İçerik yüklenirken bir hata oluştu." ? (
                        <>
                            <div className="content-area" aria-live="polite">
                                {readerContent[currentReadingBook.currentPageInExcerpt]}
                            </div>
                            <div className="pagination-controls">
                                <button
                                    onClick={() => handlePageChange(currentReadingBook.id, currentReadingBook.currentPageInExcerpt - 1)}
                                    disabled={currentReadingBook.currentPageInExcerpt === 0}
                                    className="action-button"
                                    aria-label="Önceki sayfa"
                                >
                                    &lt; Önceki Sayfa
                                </button>
                                <span>Sayfa {currentReadingBook.currentPageInExcerpt + 1} / {currentReadingBook.totalPagesInExcerpt || readerContent.length}</span>
                                <button
                                    onClick={() => handlePageChange(currentReadingBook.id, currentReadingBook.currentPageInExcerpt + 1)}
                                    disabled={currentReadingBook.currentPageInExcerpt >= (currentReadingBook.totalPagesInExcerpt || readerContent.length) -1 }
                                    className="action-button"
                                    aria-label="Sonraki sayfa"
                                >
                                    Sonraki Sayfa &gt;
                                </button>
                            </div>
                        </>
                    ) : (
                        <p className="error">{readerContent[0] || "Bu kitap için içerik bulunamadı veya yüklenemedi."}</p>
                    )}
                </div>
            )}
        </div>
    );
};
// --- End of BookReaderPage Component ---


// --- YouTubeStrategistPage Component ---
interface YouTubeStrategistPageProps {
    genAI: GoogleGenAI | null;
    setError: (error: string | null) => void;
}
const YouTubeStrategistPage: React.FC<YouTubeStrategistPageProps> = ({ genAI, setError }) => {
    const [nicheQuery, setNicheQuery] = useState<string>('Evde bitki yetiştirme');
    const [nicheAnalysisResult, setNicheAnalysisResult] = useState<NicheAnalysis | null>(null);
    const [isAnalyzingNiche, setIsAnalyzingNiche] = useState<boolean>(false);

    const [marketResearchLoading, setMarketResearchLoading] = useState<boolean>(false);
    const [marketResearchResult, setMarketResearchResult] = useState<MarketResearchData | null>(null);


    const [videoTopic, setVideoTopic] = useState<string>('');
    const [videoType, setVideoType] = useState<'reels' | 'long'>('long');
    const [videoTone, setVideoTone] = useState<string>('Bilgilendirici ve eğlenceli');
    const [specificFocus, setSpecificFocus] = useState<string>('');
    const [videoBlueprintResult, setVideoBlueprintResult] = useState<VideoBlueprint | null>(null);
    const [isGeneratingBlueprint, setIsGeneratingBlueprint] = useState<boolean>(false);

    const handleNicheAnalysis = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!nicheQuery.trim()) { setError("Analiz edilecek nişi girin."); return; }
        setIsAnalyzingNiche(true); setError(null); setNicheAnalysisResult(null); setVideoBlueprintResult(null); setMarketResearchResult(null);

        const prompt = `Belirtilen YouTube nişi ("${nicheQuery}") için kapsamlı bir pazar araştırması ve analiz yap. Google Search aracını kullanarak güncel trendleri, popüler alt konuları ve hedef kitle içgörülerini topla. Cevabını Türkçe ve JSON formatında (NicheAnalysis arayüzüne uygun) ver. Analizin şunları içermeli:
        1.  nicheSummary: Nişin genel bir özeti ve potansiyeli (metin).
        2.  popularSubTopics: Bu nişte popüler olan 3-5 alt konu başlığından oluşan bir metin DİZİSİ/LİSTESİ (string[] - metin DİZİSİ/LİSTESİ olmalı).
        3.  targetAudienceInsights: Hedef kitlenin genel demografisi, ilgi alanları ve arama niyetleri hakkında çıkarımlar (metin).
        4.  contentOpportunities: Bu nişte öne çıkabilecek 2-3 potansiyel içerik fırsatı veya benzersiz açıdan oluşan bir metin DİZİSİ/LİSTESİ (string[] - metin DİZİSİ/LİSTESİ olmalı).
        5.  keywords: Niş ile ilgili önemli anahtar kelimelerden oluşan bir metin DİZİSİ/LİSTESİ (string[] - metin DİZİSİ/LİSTESİ olmalı), (SEO için).`;
        try {
            const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { tools: [{ googleSearch: {} }], safetySettings: defaultSafetySettings } });
            const analysis = parseJsonResponse<NicheAnalysis>(response.text);
            if (analysis) {
                setNicheAnalysisResult(analysis);
                if (analysis.popularSubTopics && Array.isArray(analysis.popularSubTopics) && analysis.popularSubTopics.length > 0) {
                    setVideoTopic(analysis.popularSubTopics[0]);
                } else {
                    setVideoTopic(nicheQuery);
                     if (analysis.popularSubTopics !== undefined && !Array.isArray(analysis.popularSubTopics)) {
                        console.warn("NicheAnalysis: popularSubTopics bir dizi olarak dönmedi.", analysis.popularSubTopics);
                    }
                }
            } else {
                setError("Niş analizi sonucu ayrıştırılamadı veya hatalı: " + response.text?.substring(0, 300));
            }
        } catch (err: any) { setError(`Niş analizi hatası: ${err.message}`); }
        finally { setIsAnalyzingNiche(false); }
    };
    
    const handleMarketResearch = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!nicheQuery.trim()) { setError("Araştırılacak nişi girin."); return; }
        setMarketResearchLoading(true); setError(null); setMarketResearchResult(null);

        const prompt = `"${nicheQuery}" nişi için detaylı bir YouTube piyasa araştırması yap. Google Search aracını kullanarak aşağıdaki bilgileri topla ve Türkçe JSON formatında (MarketResearchData arayüzüne uygun) yanıt ver:
        1.  analyzedNiche: Analiz edilen niş (string).
        2.  highlyViewedVideos: Bu nişte YouTube ve TikTok gibi platformlarda en çok izlenen 3-5 video örneği. Her örnek için: title (string), platform (string, örn: "YouTube", "TikTok"), views (string, örn: "1.5M izlenme", "700K beğeni"), link (string, videoya direkt bağlantı), notes (string, AI'nın video hakkındaki kısa gözlemleri, neden popüler olduğu gibi).
        3.  platformAnalysis: Bu nişteki içeriğin YouTube, TikTok ve Instagram Reels gibi platformlardaki dağılımı ve popülerliği hakkında bir analiz. Her platform için: platformName (string), contentVolume ('high', 'medium', 'low', 'unknown' - içerik hacmi), audienceEngagement ('high', 'medium', 'low', 'unknown' - kitle etkileşimi), notes (string, platforma özel gözlemler).
        4.  generalObservations: Nişin genel durumu, rekabet seviyesi, doygunluk, trendler ve içerik üreticileri için potansiyel hakkında genel gözlemler (string).
        5.  dataSourcesUsed: Kullandığın anahtar kelimeler veya arama sorguları gibi bilgiler (string[]).
        Mümkün olduğunca güncel (son 6-12 ay) ve popüler verilere odaklan. Bağlantıların geçerli ve erişilebilir olmasına dikkat et.`;

        try {
            const response = await genAI.models.generateContent({ model: API_MODEL, contents: prompt, config: { tools: [{ googleSearch: {} }], safetySettings: defaultSafetySettings } });
            const researchData = parseJsonResponse<MarketResearchData>(response.text);
            if (researchData && researchData.analyzedNiche) {
                setMarketResearchResult(researchData);
            } else {
                setError("Piyasa araştırması sonucu ayrıştırılamadı veya hatalı: " + response.text?.substring(0, 500));
            }
        } catch (err: any) { setError(`Piyasa araştırması hatası: ${err.message}`); }
        finally { setMarketResearchLoading(false); }
    };


    const handleBlueprintGeneration = async () => {
        if (!genAI) { setError("API istemcisi başlatılamadı."); return; }
        if (!videoTopic.trim()) { setError("Video konusunu girin."); return; }
        setIsGeneratingBlueprint(true); setError(null); setVideoBlueprintResult(null);

        const videoTypeDescription = videoType === 'reels' ? "kısa dikey video (Reels/Shorts, yaklaşık 15-60 saniye)" : "uzun formatlı video (yaklaşık 5-15 dakika)";
        
        let blueprintPrompt = `Aşağıdaki bilgilere dayanarak bir YouTube video içerik üretim planı oluştur. Cevabını Türkçe ve JSON formatında (VideoBlueprint arayüzüne uygun) ver:\n
        - Niş/Ana Konu: "${videoTopic}"
        - Video Türü: ${videoTypeDescription}
        - Video Tonu: "${videoTone}"
        - Ek Odak Noktaları (isteğe bağlı): "${specificFocus || 'Yok'}"
        - Araştırılan Niş: "${nicheAnalysisResult?.nicheSummary || marketResearchResult?.analyzedNiche || nicheQuery}"
        
        Plan şunları içermelidir:
        1.  generatedForNiche: Kullanıcının girdiği orijinal niş sorgusu (metin).
        2.  videoType: "${videoType}".
        3.  videoTone: Belirtilen video tonu ("${videoTone}").
        4.  titleSuggestions: SEO dostu ve dikkat çekici 3-5 video başlığı (string dizisi).
        5.  descriptionDraft: Anahtar kelimeler içeren, videoyu özetleyen, (uzun videolar için zaman damgaları önerebilir) ve CTA içeren bir açıklama (metin).
        6.  tagsKeywords: Video için uygun etiketler ve anahtar kelimeler (string dizisi).
        7.  ${videoType === 'reels' ? 
            `storyboard: Sahne sahne döküm (StoryboardScene[] dizisi, her sahne için: sceneNumber (sayı), durationSeconds (metin, örn: "3-5 saniye"), visualDescription (metin), onScreenText (metin, varsa), voiceoverScript (metin, varsa), soundSuggestion (metin, müzik/efekt önerisi), brollSuggestions (BrollSuggestion[] dizisi, her öneri için: description (metin, örn: "doğada yürüyen kişi"), searchLinks (BrollSuggestionLink[] dizisi, her link için: siteName (string, örn: "Pexels"), url (string, örn: "https://www.pexels.com/search/URL_ENCODED_ARAMA_TERIMI/")))). En az 5-7 sahne.` :
            `scriptSegments: Bölümlere ayrılmış senaryo (ScriptSegment[] dizisi, her bölüm için: segmentTitle (metin), durationMinutes (metin, örn: "1-2 dakika"), visualIdeas (metin), voiceoverScript (metin), brollSuggestions (BrollSuggestion[] dizisi, yukarıdaki gibi)). Giriş, en az 2-3 ana bölüm ve sonuç içermeli.`
           }
        8.  ${videoType === 'long' ? `fullVoiceoverScript: Uzun video için tüm seslendirme metni (metin).` : ''}
        9.  ${videoType === 'long' ? `fullSubtitleScript: Uzun video için tüm altyazı metni (metin, seslendirmeden türetilmiş).` : ''}
        10. thumbnailConcepts: 2-3 farklı, detaylı kapak fotoğrafı konsepti (ThumbnailConcept[] dizisi, her konsept için: conceptNumber (sayı), description (metin), suggestedElements (önerilen görsel öğelerin bir metin DİZİSİ/LİSTESİ, örn: ["Parlak başlık", "Ürünün fotoğrafı"])). Bu sadece bir açıklama olmalı, görsel değil.
        11. aiToolSuggestions: { thumbnailPrompts: string[] (her thumbnailConcept için bir adet metinden-görsele AI prompt'u), voiceoverNotes: string (seslendirme metninin TTS ile kullanımı hakkında notlar), visualPromptsForScenes?: { sceneNumber?: number, sceneDescription: string, promptSuggestion: string }[] (storyboard/script segmentlerindeki bazı önemli sahneler için metinden-görsele/videoya AI prompt fikirleri, opsiyonel) }.
        12. soundtrackSuggestion: Videonun genel atmosferine uygun telifsiz müzik türü veya genel ses efekti önerileri (metin).
        13. potentialInteractionAssessment: Niş araştırmasına ve planlanan içeriğe dayanarak, videonun potansiyel etkileşimi hakkında niteliksel bir değerlendirme (metin, kesin izlenme sayısı değil, genel popülerlik/trend yorumu).

        B-roll önerileri için Pexels (https://www.pexels.com/search/QUERY/) ve Pixabay (https://pixabay.com/videos/search/QUERY/) gibi ücretsiz stok video sitelerine yönelik arama URL'leri oluştur. QUERY kısmını URL kodlu arama terimi ile değiştir.
        JSON formatının kesinlikle doğru olduğundan emin ol: tüm anahtarlar ve metin değerleri çift tırnak (" ") içinde olmalı, listeler köşeli parantez ([ ]) ve objeler kıvırcık parantez ({ }) içinde olmalı, elemanlar virgülle (,) ayrılmalı ve son elemandan sonra virgül olmamalıdır. Metinlerdeki özel karakterler (örn: çift tırnak) JSON için doğru şekilde escape edilmelidir.
        Bu plan, bir video prodüksiyonu için gerekli tüm metinleri, fikirleri, stratejileri ve diğer AI araçları için prompt önerilerini içermelidir. Kullanıcının bu planı alıp kendi video düzenleme ve AI araçlarıyla videoyu/görselleri oluşturacağını varsay.
        `;
        
        try {
            const response = await genAI.models.generateContent({ model: API_MODEL, contents: blueprintPrompt, config: { responseMimeType: "application/json", safetySettings: defaultSafetySettings } });
            const blueprint = parseJsonResponse<VideoBlueprint>(response.text);
             if (blueprint && blueprint.titleSuggestions && blueprint.descriptionDraft) {
                blueprint.generatedForNiche = nicheAnalysisResult?.nicheSummary || marketResearchResult?.analyzedNiche || nicheQuery;
                blueprint.videoType = videoType;
                blueprint.videoTone = videoTone;
                setVideoBlueprintResult(blueprint);
            } else {
                setError("Video üretim planı sonucu ayrıştırılamadı veya eksik veri içeriyor: " + response.text?.substring(0, 500));
            }
        } catch (err: any) { setError(`Video üretim planı oluşturma hatası: ${err.message}`); }
        finally { setIsGeneratingBlueprint(false); }
    };
    
    return (
        <div className="page-content strategist-page">
            <h2>AI YouTube Video Stratejisti ve İçerik Üretim Planlayıcısı</h2>
            <p>Bir YouTube nişi belirleyin, AI'nın pazar araştırması yapmasını sağlayın ve ardından seçtiğiniz video türü için kapsamlı bir içerik üretim planı oluşturun.</p>
            <p className="warning-text"><strong>Önemli Not:</strong> Bu araç size doğrudan bir video dosyası (.mp4) veya kapak fotoğrafı dosyası **oluşturmaz**. Bunun yerine, video prodüksiyonunuz için gereken tüm stratejik planı, metinleri, fikirleri, yapısal çerçeveyi ve diğer AI araçları için prompt önerilerini detaylı bir şekilde sunar.</p>

            <div className="page-section niche-research-section">
                <h3>1. Adım: Niş Girişi</h3>
                <div className="form-container single-column-form">
                    <div className="form-group">
                        <label htmlFor="nicheQuery">Araştırılacak YouTube Nişi:</label>
                        <input
                            type="text"
                            id="nicheQuery"
                            value={nicheQuery}
                            onChange={(e) => { setNicheQuery(e.target.value); setNicheAnalysisResult(null); setMarketResearchResult(null); setVideoBlueprintResult(null); }}
                            placeholder="Örn: Sürdürülebilir yaşam, Python programlama dersleri"
                            aria-label="Araştırılacak YouTube Nişi"
                        />
                    </div>
                    <div className="action-buttons-group">
                        <button onClick={handleNicheAnalysis} className="submit-button" disabled={isAnalyzingNiche || !nicheQuery.trim()} aria-label="Niş Analizi Yap">
                            {isAnalyzingNiche ? 'Niş Analiz Ediliyor...' : '1A. Niş Analizi Yap'}
                        </button>
                        <button onClick={handleMarketResearch} className="submit-button" disabled={marketResearchLoading || !nicheQuery.trim()} aria-label="Piyasayı Araştır ve Analiz Et">
                            {marketResearchLoading ? 'Piyasa Araştırılıyor...' : '1B. Piyasayı Araştır'}
                        </button>
                    </div>
                </div>

                {isAnalyzingNiche && <div className="loading" role="status" aria-live="polite">Niş analizi yapılıyor, lütfen bekleyin...</div>}
                {marketResearchLoading && <div className="loading" role="status" aria-live="polite">Piyasa araştırması yapılıyor, lütfen bekleyin...</div>}

                {marketResearchResult && (
                    <details open className="output-section market-research-section">
                        <summary><h4><span role="img" aria-label="chart decreasing">📉</span> Piyasa Araştırması ve Rakip Analizi Sonuçları (Genişlet/Daralt)</h4></summary>
                        <div className="output-subsection">
                            <strong>Analiz Edilen Niş:</strong> {marketResearchResult.analyzedNiche}
                        </div>
                        {marketResearchResult.highlyViewedVideos && marketResearchResult.highlyViewedVideos.length > 0 && (
                            <div className="output-subsection">
                                <strong>Popüler Video Örnekleri:</strong>
                                <ul>
                                    {marketResearchResult.highlyViewedVideos.map((video, i) => (
                                        <li key={`video-${i}`} className="video-example-item">
                                            <strong>Başlık:</strong> {video.title} <br />
                                            {video.platform && <><strong>Platform:</strong> {video.platform} </>}
                                            {video.views && <><strong>İzlenme/Beğeni:</strong> {video.views} </>}
                                            {video.link && <><a href={video.link} target="_blank" rel="noopener noreferrer" aria-label={`${video.title} videosuna git`}>Videoya Git</a> </>}
                                            {video.notes && <><br /><em>Notlar: {video.notes}</em></> }
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {marketResearchResult.platformAnalysis && marketResearchResult.platformAnalysis.length > 0 && (
                             <div className="output-subsection">
                                <strong>Platform Dağılımı ve Popülerlik:</strong>
                                <ul>
                                    {marketResearchResult.platformAnalysis.map((platform, i) => (
                                        <li key={`platform-${i}`} className="platform-distribution-item">
                                            <strong>{platform.platformName}:</strong> İçerik Hacmi: {platform.contentVolume}, Kitle Etkileşimi: {platform.audienceEngagement}
                                            {platform.notes && <><br /><em>Notlar: {platform.notes}</em></>}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {marketResearchResult.generalObservations && (
                            <div className="output-subsection">
                                <strong>Genel Gözlemler:</strong>
                                <p className="ai-generated-text small-text">{marketResearchResult.generalObservations}</p>
                            </div>
                        )}
                         {marketResearchResult.dataSourcesUsed && marketResearchResult.dataSourcesUsed.length > 0 && (
                            <div className="output-subsection">
                                <strong>Kullanılan Veri Kaynakları/Aramalar:</strong>
                                <p className="small-text">{marketResearchResult.dataSourcesUsed.join(', ')}</p>
                            </div>
                        )}
                    </details>
                )}


                {nicheAnalysisResult && (
                    <details open className="output-section">
                        <summary><h4><span role="img" aria-label="magnifying glass">🔍</span> Niş Analizi Sonuçları (Genişlet/Daralt)</h4></summary>
                        <div className="output-subsection">
                            <strong>Niş Özeti:</strong>
                            <p className="ai-generated-text small-text">{nicheAnalysisResult.nicheSummary}</p>
                        </div>
                        <div className="output-subsection">
                            <strong>Popüler Alt Konular (Video konusu olarak seçmek için tıklayın):</strong>
                            {Array.isArray(nicheAnalysisResult.popularSubTopics) && nicheAnalysisResult.popularSubTopics.length > 0 ? (
                                <ul>{nicheAnalysisResult.popularSubTopics.map((topic, i) => 
                                    <li key={i} className="clickable-topic" onClick={() => setVideoTopic(topic)} aria-label={`${topic} konusunu video konusu olarak ayarla`} role="button" tabIndex={0} onKeyDown={(e)=> e.key === 'Enter' && setVideoTopic(topic)}>
                                        {topic}
                                    </li>
                                )}</ul>
                            ) : <p className="small-text">Popüler alt konular bulunamadı veya AI tarafından sağlanmadı.</p>}
                        </div>
                        <div className="output-subsection">
                            <strong>Hedef Kitle İçgörüleri:</strong>
                            <p className="ai-generated-text small-text">{nicheAnalysisResult.targetAudienceInsights}</p>
                        </div>
                         <div className="output-subsection">
                            <strong>İçerik Fırsatları:</strong>
                            {Array.isArray(nicheAnalysisResult.contentOpportunities) && nicheAnalysisResult.contentOpportunities.length > 0 ? (
                                <ul>{nicheAnalysisResult.contentOpportunities.map((opp, i) => <li key={i}>{opp}</li>)}</ul>
                             ) : <p className="small-text">İçerik fırsatları bulunamadı veya AI tarafından sağlanmadı.</p>}
                        </div>
                        <div className="output-subsection">
                            <strong>Anahtar Kelimeler:</strong>
                             <p className="small-text">{Array.isArray(nicheAnalysisResult.keywords) && nicheAnalysisResult.keywords.length > 0 ? nicheAnalysisResult.keywords.join(', ') : 'Anahtar kelimeler bulunamadı veya AI tarafından sağlanmadı.'}</p>
                        </div>
                    </details>
                )}
            </div>

            {(nicheAnalysisResult || marketResearchResult || videoTopic) && (
                 <div className="page-section blueprint-generation-section">
                    <h3>2. Adım: Video İçerik Üretim Planı Oluşturma</h3>
                    <div className="form-container single-column-form">
                        <div className="form-group">
                            <label htmlFor="videoTopic">Video Konusu/Ana Fikri:</label>
                            <input
                                type="text"
                                id="videoTopic"
                                value={videoTopic}
                                onChange={(e) => setVideoTopic(e.target.value)}
                                placeholder="Niş analizinden bir konu seçin veya yeni bir fikir girin"
                                aria-label="Video Konusu veya Ana Fikri"
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="videoType">Video Türü:</label>
                            <select id="videoType" value={videoType} onChange={(e) => setVideoType(e.target.value as 'reels' | 'long')} aria-label="Video Türü">
                                <option value="long">Uzun Format Video (Örn: 5-15 dk)</option>
                                <option value="reels">Reels/Shorts Videosu</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label htmlFor="videoTone">Video Tonu:</label>
                            <input
                                type="text"
                                id="videoTone"
                                value={videoTone}
                                onChange={(e) => setVideoTone(e.target.value)}
                                placeholder="Örn: Komik, Ciddi, Eğitici, İlham Verici"
                                aria-label="Video Tonu"
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="specificFocus">Belirli Bir Odak Noktası (isteğe bağlı):</label>
                            <textarea
                                id="specificFocus"
                                value={specificFocus}
                                onChange={(e) => setSpecificFocus(e.target.value)}
                                rows={2}
                                placeholder="Videoda vurgulanmasını istediğiniz özel bir nokta veya mesaj"
                                aria-label="Videodaki Belirli Odak Noktası"
                            />
                        </div>
                        <button onClick={handleBlueprintGeneration} className="submit-button" disabled={isGeneratingBlueprint || !videoTopic.trim()} aria-label="İçerik Üretim Planı Oluştur">
                            {isGeneratingBlueprint ? 'Plan Oluşturuluyor...' : 'İçerik Üretim Planı Oluştur'}
                        </button>
                    </div>
                    {isGeneratingBlueprint && <div className="loading" role="status" aria-live="polite">Video üretim planı oluşturuluyor...</div>}
                 </div>
            )}

            {videoBlueprintResult && (
                <div className="page-section blueprint-output-section">
                    <h3><span role="img" aria-label="clapper board">🎬</span> Video İçerik Üretim Planı Sonucu:</h3>
                    <details open className="output-subsection">
                        <summary><strong>Genel Bilgiler</strong></summary>
                        <p><strong>Niş:</strong> {videoBlueprintResult.generatedForNiche}</p>
                        <p><strong>Video Türü:</strong> {videoBlueprintResult.videoType === 'reels' ? 'Reels/Shorts' : 'Uzun Format'}</p>
                        <p><strong>Video Tonu:</strong> {videoBlueprintResult.videoTone || "Belirtilmemiş"}</p>
                    </details>
                    
                    <details className="output-subsection">
                        <summary><strong>Başlık Önerileri</strong></summary>
                        <ul>{videoBlueprintResult.titleSuggestions.map((title, i) => <li key={i}>{title} <button className="copy-button-small" onClick={() => copyToClipboard(title)} aria-label={`${title} başlığını kopyala`}>Kopyala</button></li>)}</ul>
                    </details>
                    <details className="output-subsection">
                        <summary><strong>Açıklama Taslağı</strong></summary>
                        <button className="copy-button-small" onClick={() => copyToClipboard(videoBlueprintResult.descriptionDraft)} aria-label="Açıklama taslağını kopyala">Kopyala</button>
                        <div className="ai-generated-text">{videoBlueprintResult.descriptionDraft}</div>
                    </details>
                    <details className="output-subsection">
                        <summary><strong>Etiketler/Anahtar Kelimeler</strong></summary>
                        <p>{videoBlueprintResult.tagsKeywords.join(', ')}</p>
                    </details>

                    {videoBlueprintResult.storyboard && videoBlueprintResult.videoType === 'reels' && (
                        <details className="output-subsection">
                            <summary><strong>Reels/Shorts Storyboard</strong></summary>
                            {videoBlueprintResult.storyboard.map(scene => (
                                <div key={scene.sceneNumber} className="storyboard-scene">
                                    <strong>Sahne {scene.sceneNumber}:</strong> (Süre: {scene.durationSeconds || 'Belirtilmemiş'})
                                    <p><strong>Görsel:</strong> {scene.visualDescription}</p>
                                    {scene.onScreenText && <p><strong>Ekran Metni:</strong> {scene.onScreenText}</p>}
                                    {scene.voiceoverScript && <p><strong>Seslendirme:</strong> {scene.voiceoverScript}</p>}
                                    {scene.soundSuggestion && <p><strong>Ses/Müzik:</strong> {scene.soundSuggestion}</p>}
                                    {scene.brollSuggestions && scene.brollSuggestions.length > 0 && (
                                        <div className="broll-suggestions-list">
                                            <strong>B-Roll/Stok Video Önerileri:</strong>
                                            {scene.brollSuggestions.map((broll, bIndex) => (
                                                <div key={`broll-${scene.sceneNumber}-${bIndex}`} className="broll-item">
                                                    <p>{broll.description}</p>
                                                    {broll.searchLinks && broll.searchLinks.length > 0 && (
                                                        <div className="search-links-list">
                                                            {broll.searchLinks.map((link, lIndex) => (
                                                                <a key={`link-${scene.sceneNumber}-${bIndex}-${lIndex}`} href={link.url} target="_blank" rel="noopener noreferrer" className="search-link-button" aria-label={`${broll.description} için ${link.siteName}'da ara`}>
                                                                    {link.siteName}'da Ara
                                                                </a>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </details>
                    )}

                    {videoBlueprintResult.scriptSegments && videoBlueprintResult.videoType === 'long' && (
                         <details className="output-subsection">
                            <summary><strong>Uzun Video Senaryo Akışı</strong></summary>
                            {videoBlueprintResult.scriptSegments.map((segment, i) => (
                                <div key={i} className="script-segment">
                                    <strong>{segment.segmentTitle}</strong> (Süre: {segment.durationMinutes || 'Belirtilmemiş'})
                                    <p><strong>Görsel Fikirleri:</strong> {segment.visualIdeas}</p>
                                    <p><strong>Seslendirme Metni:</strong> {segment.voiceoverScript}</p>
                                    {segment.brollSuggestions && segment.brollSuggestions.length > 0 && (
                                        <div className="broll-suggestions-list">
                                            <strong>B-Roll/Stok Video Önerileri:</strong>
                                            {segment.brollSuggestions.map((broll, bIndex) => (
                                                <div key={`broll-long-${i}-${bIndex}`} className="broll-item">
                                                    <p>{broll.description}</p>
                                                    {broll.searchLinks && broll.searchLinks.length > 0 && (
                                                        <div className="search-links-list">
                                                            {broll.searchLinks.map((link, lIndex) => (
                                                                <a key={`link-long-${i}-${bIndex}-${lIndex}`} href={link.url} target="_blank" rel="noopener noreferrer" className="search-link-button" aria-label={`${broll.description} için ${link.siteName}'da ara`}>
                                                                    {link.siteName}'da Ara
                                                                </a>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </details>
                    )}
                     {videoBlueprintResult.fullVoiceoverScript && videoBlueprintResult.videoType === 'long' && (
                        <details className="output-subsection">
                            <summary><strong>Tam Seslendirme Metni</strong></summary>
                            <button className="copy-button-small" onClick={() => copyToClipboard(videoBlueprintResult.fullVoiceoverScript!)} aria-label="Tam seslendirme metnini kopyala">Kopyala</button>
                            <div className="ai-generated-text full-script-section">{videoBlueprintResult.fullVoiceoverScript}</div>
                        </details>
                    )}
                     {videoBlueprintResult.fullSubtitleScript && videoBlueprintResult.videoType === 'long' && (
                        <details className="output-subsection">
                            <summary><strong>Tam Altyazı Metni</strong></summary>
                            <button className="copy-button-small" onClick={() => copyToClipboard(videoBlueprintResult.fullSubtitleScript!)} aria-label="Tam altyazı metnini kopyala">Kopyala</button>
                            <div className="ai-generated-text full-script-section">{videoBlueprintResult.fullSubtitleScript}</div>
                        </details>
                    )}

                    <details className="output-subsection thumbnail-concept-section">
                        <summary><strong>Kapak Fotoğrafı Konseptleri</strong></summary>
                        {videoBlueprintResult.thumbnailConcepts.map(concept => (
                            <div key={concept.conceptNumber} className="thumbnail-concept-item">
                                <strong>Konsept {concept.conceptNumber}:</strong>
                                <p>{concept.description}</p>
                                {Array.isArray(concept.suggestedElements) && concept.suggestedElements.length > 0 ? (
                                    <p><em>Önerilen Öğeler: {concept.suggestedElements.join(', ')}</em></p>
                                ) : concept.suggestedElements && typeof concept.suggestedElements === 'string' ? (
                                    <p><em>Önerilen Öğeler: {concept.suggestedElements}</em></p>
                                ) : (
                                    <p><em>Önerilen Öğeler: Yok veya belirtilmemiş.</em></p>
                                )}
                            </div>
                        ))}
                    </details>

                    <details className="output-subsection production-tools-section">
                        <summary><strong>3. Adım: Prodüksiyon Araçları ve Kaynakları (Genişlet/Daralt)</strong></summary>
                        {videoBlueprintResult.aiToolSuggestions && (
                            <div className="ai-tool-suggestions-section">
                                {videoBlueprintResult.aiToolSuggestions.thumbnailPrompts && videoBlueprintResult.aiToolSuggestions.thumbnailPrompts.length > 0 && (
                                    <div className="tool-suggestion-item">
                                        <strong>Kapak Fotoğrafı için Metinden-Görsele AI Prompt'ları:</strong>
                                        <ul className="thumbnail-prompts-list">
                                            {videoBlueprintResult.aiToolSuggestions.thumbnailPrompts.map((prompt, i) => (
                                                <li key={`thumb-prompt-${i}`}>
                                                    "{prompt}"
                                                    <button className="copy-button-small" onClick={() => copyToClipboard(prompt)} aria-label={`Kapak fotoğrafı prompt ${i+1} kopyala`}>Kopyala</button>
                                                </li>
                                            ))}
                                        </ul>
                                        <p className="small-text">Bu prompt'ları DALL-E, Midjourney, Stable Diffusion gibi araçlarda kullanabilirsiniz.</p>
                                    </div>
                                )}
                                {videoBlueprintResult.aiToolSuggestions.voiceoverNotes && (
                                    <div className="tool-suggestion-item">
                                        <strong>Seslendirme (Metinden-Sese - TTS):</strong>
                                        <p className="ai-generated-text small-text">{videoBlueprintResult.aiToolSuggestions.voiceoverNotes}</p>
                                        <p className="small-text">Bu metni [Popüler bir TTS aracı adı] veya benzeri bir yapay zeka aracıyla seslendirebilirsiniz.</p>
                                    </div>
                                )}
                                {videoBlueprintResult.aiToolSuggestions.visualPromptsForScenes && videoBlueprintResult.aiToolSuggestions.visualPromptsForScenes.length > 0 && (
                                    <div className="tool-suggestion-item">
                                        <strong>Video Sahneleri için Görselleştirme Prompt Fikirleri (Deneysel):</strong>
                                        <ul className="scene-visual-prompts-list">
                                            {videoBlueprintResult.aiToolSuggestions.visualPromptsForScenes.map((scenePrompt, i) => (
                                                <li key={`scene-prompt-${i}`}>
                                                    <strong>{scenePrompt.sceneNumber ? `Sahne ${scenePrompt.sceneNumber}` : 'Genel Sahne' } ({scenePrompt.sceneDescription.substring(0,50)}...):</strong><br/>
                                                    Prompt Önerisi: "{scenePrompt.promptSuggestion}"
                                                    <button className="copy-button-small" onClick={() => copyToClipboard(scenePrompt.promptSuggestion)} aria-label={`Sahne prompt ${i+1} kopyala`}>Kopyala</button>
                                                </li>
                                            ))}
                                        </ul>
                                        <p className="small-text">Bu prompt'lar, metinden-görsele veya metinden-videoya AI araçları için ilham verebilir.</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </details>
                    
                    {videoBlueprintResult.soundtrackSuggestion &&
                        <details className="output-subsection">
                            <summary><strong>Genel Müzik/Ses Önerisi</strong></summary>
                            <p>{videoBlueprintResult.soundtrackSuggestion}</p>
                        </details>
                    }
                    <details className="output-subsection">
                        <summary><strong>Potansiyel Etkileşim Değerlendirmesi</strong></summary>
                        <p className="ai-generated-text small-text">{videoBlueprintResult.potentialInteractionAssessment}</p>
                    </details>
                </div>
            )}
        </div>
    );
};
// --- End of YouTubeStrategistPage Component ---

// --- Main App Component ---
const App: React.FC = () => {
  const [error, setError] = useState<string | null>(null);
  const [initializationError] = useState<string | null>(globalInitializationError);
  const [currentPage, setCurrentPage] = useState<'planner' | 'priceAnalysis' | 'healthCheck' | 'bookReader' | 'youtubeStrategist'>('planner');
  
  const renderPageContent = () => {
    if (initializationError) return <div className="error global-error" role="alert"><strong>API Başlatma Hatası:</strong> {initializationError}</div>;
    
    const pageProps = { genAI, setError };

    switch (currentPage) {
      case 'planner': return <PlannerPage {...pageProps} />;
      case 'healthCheck': return <HealthCheckPage {...pageProps} />;
      case 'priceAnalysis': return <PriceAnalysisPage {...pageProps} />;
      case 'bookReader': return <BookReaderPage {...pageProps} />;
      case 'youtubeStrategist': return <YouTubeStrategistPage {...pageProps} />;
      default: 
        const exhaustiveCheck: never = currentPage; 
        return <div>Bilinmeyen sayfa. Lütfen navigasyondan bir seçim yapın.</div>;
    }
  };

  return (
    <>
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="container">
        <h1 className="app-title">AI Destekli Çok Amaçlı Asistan</h1>
        {error && 
            <div className="error global-error" role="alert">
                <strong>Hata:</strong> {error} 
                <button onClick={() => setError(null)} className="close-error-button" aria-label="Hata mesajını kapat">Kapat</button>
            </div>
        }
        {renderPageContent()}
      </div>
      <footer className="app-footer">
        <p>© {new Date().getFullYear()} AI Asistan Uygulaması. Tüm hakları saklıdır.</p>
        <p className="footer-disclaimer">Bu uygulama demo amaçlıdır. Sağlanan bilgiler tıbbi veya profesyonel tavsiye niteliği taşımaz.</p>
      </footer>
    </>
  );
};
// --- End of Main App Component ---

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<React.StrictMode><App /></React.StrictMode>);
} else {
  console.error("Root element 'root' bulunamadı.");
}
