"use client"

import { useState, useCallback, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Music, Brain, Trophy, Loader2, RefreshCw, CheckCircle2, XCircle, BarChart3, ArrowLeft } from "lucide-react"

interface Prediction {
    artist_id: string
    artist_name: string
    confidence: number
    votes: number
    rank: number
}

interface SongData {
    track_id: string
    song_title: string
    year: number | null
    duration: number | null
    predictions: Prediction[]
    actual: {
        artist_id: string
        artist_name: string
    }
}

type GameState = "start" | "playing" | "result" | "finished"
type View = "game" | "stats"

interface Stats {
    total_songs_in_database: number
    total_artists_in_database: number
    total_training_examples: number
    learnable_artists_count: number
    learnable_songs_count: number
    min_training_examples_threshold: number
    playable_percentage: number
}

const TOTAL_ROUNDS = 10
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"

function shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
}

export function MLGuessingGame() {
    const [view, setView] = useState<View>("game")
    const [gameState, setGameState] = useState<GameState>("start")
    const [currentRound, setCurrentRound] = useState(1)
    const [score, setScore] = useState(0)
    const [currentSong, setCurrentSong] = useState<SongData | null>(null)
    const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [roundHistory, setRoundHistory] = useState<
        Array<{
            song: string
            userCorrect: boolean
            modelCorrect: boolean
            modelPrediction: string
            actualArtist: string
            confidence: number
        }>
    >([])
    const [stats, setStats] = useState<Stats | null>(null)
    const [statsLoading, setStatsLoading] = useState(false)
    const [statsError, setStatsError] = useState<string | null>(null)

    const shuffledOptions = useMemo(() => {
        if (!currentSong) return []
        return shuffleArray(currentSong.predictions.map((p) => p.artist_name))
    }, [currentSong])

    const fetchRandomSong = useCallback(async () => {
        setIsLoading(true)
        setError(null)
        try {
            const response = await fetch(`${API_BASE_URL}/random-song`)
            if (!response.ok) {
                throw new Error("Failed to fetch song")
            }
            const data: SongData = await response.json()
            setCurrentSong(data)
            setSelectedAnswer(null)
        } catch (err) {
            setError("Could not connect to the backend. Make sure the Flask server is running on port 5000.")
        } finally {
            setIsLoading(false)
        }
    }, [])

    const fetchStats = useCallback(async () => {
        setStatsLoading(true)
        setStatsError(null)
        try {
            const response = await fetch(`${API_BASE_URL}/stats`)
            if (!response.ok) {
                throw new Error("Failed to fetch stats")
            }
            const data: Stats = await response.json()
            setStats(data)
        } catch (err) {
            setStatsError("Could not connect to the backend. Make sure the Flask server is running.")
        } finally {
            setStatsLoading(false)
        }
    }, [])

    useEffect(() => {
        if (view === "stats" && !stats) {
            fetchStats()
        }
    }, [view, stats, fetchStats])

    const startGame = () => {
        setGameState("playing")
        setCurrentRound(1)
        setScore(0)
        setRoundHistory([])
        fetchRandomSong()
    }

    const handleAnswer = (answer: string) => {
        if (!currentSong || selectedAnswer) return

        setSelectedAnswer(answer)

        const modelTopPrediction = currentSong.predictions[0]?.artist_name
        const actualArtist = currentSong.actual.artist_name
        const userCorrect = answer === modelTopPrediction
        const modelCorrect = modelTopPrediction === actualArtist

        if (userCorrect) {
            setScore((prev) => prev + 1)
        }

        setRoundHistory((prev) => [
            ...prev,
            {
                song: currentSong.song_title,
                userCorrect,
                modelCorrect,
                modelPrediction: modelTopPrediction,
                actualArtist,
                confidence: currentSong.predictions[0]?.confidence || 0,
            },
        ])

        setGameState("result")
    }

    const nextRound = () => {
        if (currentRound >= TOTAL_ROUNDS) {
            setGameState("finished")
        } else {
            setCurrentRound((prev) => prev + 1)
            setGameState("playing")
            fetchRandomSong()
        }
    }

    const formatDuration = (seconds: number | null) => {
        if (!seconds) return "Unknown"
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, "0")}`
    }

    if (view === "stats") {
        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg">
                    <CardHeader className="pb-4">
                        <div className="flex items-center gap-2">
                            <Button variant="ghost" size="icon" onClick={() => setView("game")}>
                                <ArrowLeft className="w-4 h-4" />
                            </Button>
                            <div>
                                <CardTitle className="text-xl">Model Statistics</CardTitle>
                                <CardDescription>Training data and model information</CardDescription>
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {statsLoading && (
                            <div className="flex items-center justify-center py-8">
                                <Loader2 className="w-8 h-8 animate-spin text-primary" />
                            </div>
                        )}
                        {statsError && (
                            <div className="text-center py-8 space-y-4">
                                <XCircle className="w-8 h-8 mx-auto text-destructive" />
                                <p className="text-destructive text-sm">{statsError}</p>
                                <Button onClick={fetchStats} variant="outline" size="sm">
                                    <RefreshCw className="w-4 h-4 mr-2" />
                                    Retry
                                </Button>
                            </div>
                        )}
                        {stats && (
                            <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-muted/50 rounded-lg p-4">
                                        <p className="text-2xl font-bold text-primary">{stats.total_songs_in_database.toLocaleString()}</p>
                                        <p className="text-sm text-muted-foreground">Total Songs</p>
                                    </div>
                                    <div className="bg-muted/50 rounded-lg p-4">
                                        <p className="text-2xl font-bold text-primary">
                                            {stats.total_artists_in_database.toLocaleString()}
                                        </p>
                                        <p className="text-sm text-muted-foreground">Total Artists</p>
                                    </div>
                                    <div className="bg-muted/50 rounded-lg p-4">
                                        <p className="text-2xl font-bold text-primary">{stats.total_training_examples.toLocaleString()}</p>
                                        <p className="text-sm text-muted-foreground">Training Examples</p>
                                    </div>
                                    <div className="bg-muted/50 rounded-lg p-4">
                                        <p className="text-2xl font-bold text-primary">{stats.learnable_artists_count.toLocaleString()}</p>
                                        <p className="text-sm text-muted-foreground">Learnable Artists</p>
                                    </div>
                                </div>
                                <div className="bg-muted/50 rounded-lg p-4 space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm text-muted-foreground">Playable Songs</span>
                                        <span className="font-medium">{stats.learnable_songs_count.toLocaleString()}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm text-muted-foreground">Playable Percentage</span>
                                        <span className="font-medium">{stats.playable_percentage}%</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm text-muted-foreground">Min Training Threshold</span>
                                        <span className="font-medium">{stats.min_training_examples_threshold} examples</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Start Screen
    if (gameState === "start") {
        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg text-center">
                    <CardHeader className="pb-4">
                        <div className="mx-auto mb-4 w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                            <Brain className="w-8 h-8 text-primary" />
                        </div>
                        <CardTitle className="text-2xl">ML Artist Guesser</CardTitle>
                        <CardDescription className="text-base">
                            Can you predict what the machine learning model thinks?
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="bg-muted/50 rounded-lg p-4 text-left space-y-2 text-sm">
                            <p className="font-medium">How to play:</p>
                            <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                                <li>{"You'll be shown a song title"}</li>
                                <li>Guess which artist the ML model predicts</li>
                                <li>
                                    Choose from the model{"'"}s top predictions or {'"'}None of the above{'"'}
                                </li>
                                <li>Score points for matching the model{"'"}s top prediction</li>
                            </ul>
                        </div>
                        <div className="space-y-3">
                            <Button onClick={startGame} size="lg" className="w-full">
                                <Music className="w-4 h-4 mr-2" />
                                Start Game ({TOTAL_ROUNDS} Rounds)
                            </Button>
                            <Button onClick={() => setView("stats")} variant="outline" className="w-full">
                                <BarChart3 className="w-4 h-4 mr-2" />
                                View Model Stats
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Loading State
    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg text-center">
                    <CardContent className="py-12">
                        <Loader2 className="w-12 h-12 animate-spin mx-auto text-primary" />
                        <p className="mt-4 text-muted-foreground">Loading next song...</p>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Error State
    if (error) {
        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg text-center">
                    <CardContent className="py-12 space-y-4">
                        <XCircle className="w-12 h-12 mx-auto text-destructive" />
                        <p className="text-destructive">{error}</p>
                        <Button onClick={fetchRandomSong} variant="outline">
                            <RefreshCw className="w-4 h-4 mr-2" />
                            Try Again
                        </Button>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Playing State
    if (gameState === "playing" && currentSong) {
        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg">
                    <CardHeader className="pb-4">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-muted-foreground">
                                Round {currentRound} of {TOTAL_ROUNDS}
                            </span>
                            <span className="text-sm font-medium">
                                Score: {score}/{currentRound - 1}
                            </span>
                        </div>
                        <Progress value={(currentRound / TOTAL_ROUNDS) * 100} className="h-2" />
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="text-center space-y-2">
                            <p className="text-sm text-muted-foreground">Who does the model think made this song?</p>
                            <h2 className="text-xl font-bold text-balance">
                                {'"'}
                                {currentSong.song_title}
                                {'"'}
                            </h2>
                            <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
                                {currentSong.year && <span>Year: {currentSong.year}</span>}
                                {currentSong.duration && <span>Duration: {formatDuration(currentSong.duration)}</span>}
                            </div>
                        </div>

                        <div className="space-y-3">
                            {shuffledOptions.map((artist, index) => (
                                <Button
                                    key={artist}
                                    variant="outline"
                                    className="w-full justify-start h-auto py-3 px-4 bg-transparent"
                                    onClick={() => handleAnswer(artist)}
                                >
                                    <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-sm font-medium flex items-center justify-center mr-3">
                                        {index + 1}
                                    </span>
                                    {artist}
                                </Button>
                            ))}
                            <Button
                                variant="outline"
                                className="w-full justify-start h-auto py-3 px-4 border-dashed bg-transparent"
                                onClick={() => handleAnswer("none")}
                            >
                                <span className="w-6 h-6 rounded-full bg-muted text-muted-foreground text-sm font-medium flex items-center justify-center mr-3">
                                    ?
                                </span>
                                None of the above
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Result State
    if (gameState === "result" && currentSong) {
        const modelTopPrediction = currentSong.predictions[0]
        const isCorrect = selectedAnswer === modelTopPrediction?.artist_name
        const actualArtist = currentSong.actual.artist_name
        // Check if model was accurate
        const modelWasCorrect = modelTopPrediction?.artist_name === actualArtist

        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg">
                    <CardHeader className="pb-4">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-muted-foreground">
                                Round {currentRound} of {TOTAL_ROUNDS}
                            </span>
                            <span className="text-sm font-medium">
                                Score: {score}/{currentRound}
                            </span>
                        </div>
                        <Progress value={(currentRound / TOTAL_ROUNDS) * 100} className="h-2" />
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="text-center space-y-3">
                            {isCorrect ? (
                                <div className="mx-auto w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center">
                                    <CheckCircle2 className="w-8 h-8 text-green-500" />
                                </div>
                            ) : (
                                <div className="mx-auto w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center">
                                    <XCircle className="w-8 h-8 text-red-500" />
                                </div>
                            )}
                            <h2 className="text-xl font-bold">{isCorrect ? "Correct!" : "Not quite!"}</h2>
                        </div>

                        <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                            <p className="text-sm text-muted-foreground">Song:</p>
                            <p className="font-medium">
                                {'"'}
                                {currentSong.song_title}
                                {'"'}
                            </p>

                            <p className="text-sm text-muted-foreground pt-2">Model{"'"}s top prediction:</p>
                            <div className="flex items-center justify-between">
                                <p className="font-medium">{modelTopPrediction?.artist_name}</p>
                                <span className="text-sm bg-primary/10 text-primary px-2 py-1 rounded-full">
                                    {modelTopPrediction?.confidence}% confidence
                                </span>
                            </div>

                            <p className="text-sm text-muted-foreground pt-2">Actual artist:</p>
                            <p className="font-medium">{actualArtist}</p>

                            <div
                                className={`mt-3 p-2 rounded-lg flex items-center gap-2 ${modelWasCorrect ? "bg-green-500/10" : "bg-red-500/10"}`}
                            >
                                {modelWasCorrect ? (
                                    <>
                                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                                        <span className="text-sm text-green-700 dark:text-green-400">Model was correct!</span>
                                    </>
                                ) : (
                                    <>
                                        <XCircle className="w-4 h-4 text-red-500" />
                                        <span className="text-sm text-red-700 dark:text-red-400">Model was incorrect</span>
                                    </>
                                )}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <p className="text-sm font-medium text-muted-foreground">All predictions:</p>
                            {currentSong.predictions.map((pred, index) => (
                                <div
                                    key={pred.artist_id}
                                    className={`flex items-center justify-between p-2 rounded-lg ${index === 0 ? "bg-primary/5 border border-primary/20" : "bg-muted/30"
                                        }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <span className="w-5 h-5 rounded-full bg-muted text-xs flex items-center justify-center">
                                            {index + 1}
                                        </span>
                                        <span className={index === 0 ? "font-medium" : ""}>{pred.artist_name}</span>
                                    </div>
                                    <span className="text-sm text-muted-foreground">{pred.confidence}%</span>
                                </div>
                            ))}
                        </div>

                        <Button onClick={nextRound} className="w-full">
                            {currentRound >= TOTAL_ROUNDS ? "See Final Score" : "Next Round"}
                        </Button>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Finished State
    if (gameState === "finished") {
        const userPercentage = Math.round((score / TOTAL_ROUNDS) * 100)
        const modelCorrectCount = roundHistory.filter((r) => r.modelCorrect).length
        const modelPercentage = Math.round((modelCorrectCount / TOTAL_ROUNDS) * 100)

        return (
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-lg">
                    <CardHeader className="text-center pb-4">
                        <div className="mx-auto mb-4 w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
                            <Trophy className="w-10 h-10 text-primary" />
                        </div>
                        <CardTitle className="text-2xl">Game Complete!</CardTitle>
                        <CardDescription>{"Here's"} how well you predicted the ML model</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="text-center bg-muted/50 rounded-lg p-4">
                                <p className="text-sm text-muted-foreground mb-1">Your Score</p>
                                <p className="text-4xl font-bold text-primary">
                                    {score}/{TOTAL_ROUNDS}
                                </p>
                                <p className="text-muted-foreground text-sm">{userPercentage}% accuracy</p>
                            </div>
                            <div className="text-center bg-muted/50 rounded-lg p-4">
                                <p className="text-sm text-muted-foreground mb-1">Model Score</p>
                                <p className="text-4xl font-bold text-primary">
                                    {modelCorrectCount}/{TOTAL_ROUNDS}
                                </p>
                                <p className="text-muted-foreground text-sm">{modelPercentage}% accuracy</p>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <p className="text-sm font-medium text-muted-foreground">Round History:</p>
                            <div className="max-h-48 overflow-y-auto space-y-2">
                                {roundHistory.map((round, index) => (
                                    <div key={index} className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                                        <div className="flex flex-col gap-0.5">
                                            <div className="flex items-center gap-1">
                                                {round.userCorrect ? (
                                                    <CheckCircle2 className="w-3 h-3 text-green-500" />
                                                ) : (
                                                    <XCircle className="w-3 h-3 text-red-500" />
                                                )}
                                                <span className="text-xs text-muted-foreground">You</span>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                {round.modelCorrect ? (
                                                    <CheckCircle2 className="w-3 h-3 text-green-500" />
                                                ) : (
                                                    <XCircle className="w-3 h-3 text-red-500" />
                                                )}
                                                <span className="text-xs text-muted-foreground">ML</span>
                                            </div>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm truncate">{round.song}</p>
                                            <p className="text-xs text-muted-foreground">
                                                Model: {round.modelPrediction} ({round.confidence}%)
                                            </p>
                                            <p className="text-xs text-muted-foreground">Actual: {round.actualArtist}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-3">
                            <Button onClick={startGame} className="w-full">
                                <RefreshCw className="w-4 h-4 mr-2" />
                                Play Again
                            </Button>
                            <Button onClick={() => setView("stats")} variant="outline" className="w-full">
                                <BarChart3 className="w-4 h-4 mr-2" />
                                View Model Stats
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        )
    }

    return null
}
