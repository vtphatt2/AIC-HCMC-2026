# Main GUI Showcase Report

The **AIC 2026 Video Retrieval Playground** helps users find relevant moments in a video collection. Users can describe a scene, search spoken content, or describe a sequence of events. The main interface displays matching frames and lets users inspect the original video and save useful findings.

For the practical showcase, the main tab demonstrates the complete process: **enter a query → explore the results → verify a video moment → save a candidate**.

## 1. Main layout

The main tab has two main areas:

- **Left search panel:** contains the search mode, query inputs, strategy selection, filters, and search button.
- **Right results area:** displays matching frames or transcript passages, together with result counts and search-response time.

Clicking a result opens a **video viewer** over the page. Users can drag the divider to resize the left panel or collapse it using the arrow in its header, giving the results more space.

## 2. Search controls

| Feature | Location | Purpose and how to use it |
|---|---|---|
| **Frames / Transcripts** | Near the top of the left panel | Choose **Frames** to search for visual scenes, or **Transcripts** to search spoken content. Each mode displays its own search controls. |
| **Manual / Command mode** | Beside the search-mode buttons | Select **✎** for the manual form or **`>_`** for the command interface. Command mode accepts query text and commands such as `/topk 50`, `/view video`, and `/search`. |
| **Strategy** | Upper part of the manual Frames panel | Select the method used to retrieve and combine matches, such as visual search or temporal search. The description below the selector explains the selected strategy. |
| **Configuration and Tune** | Below the strategy selector | Choose a saved configuration. For configurable strategies, **Tune** opens a separate tab for adjusting search parameters. |
| **Scene description** | Search box in the left panel | Describe the scene to find, for example, “a red car driving on a road.” Click **Search** or press **Enter** in the query input. |
| **VI→EN translation** | Beside the scene-description box | Translate a Vietnamese description into English. The translation replaces the input text and can be reviewed or edited before searching. |
| **Add Temporal Step** | Above the query boxes | Add another event to a sequence. Each additional step has a description and a time-gap field in seconds after the previous step. Use a temporal strategy to search for related events in order. Unwanted steps can be removed. |
| **Top K** | Search controls below the query boxes | Set the requested number of matches. Frame search accepts values from **1 to 1,000**. The actual result count may be lower when fewer matches remain after filtering. |
| **Genre** | Beside Top K | Narrow the frame search to a selected category, or choose **All** to search across categories. |
| **Vector-search algorithm** | Gear button beside the search controls | Select an available retrieval method, such as **HNSW** or **FLAT**. The menu includes descriptions and disables unavailable options. |

In **Transcripts** mode, users enter a phrase or topic and choose a matching method:

- **Semantic:** finds passages related in meaning to the query.
- **Lexical:** searches using the words in the query.
- **Fuzzy:** supports approximate text matching, including spelling variations.

The transcript panel also provides Top K, a Topic filter when supported, and a **Search Transcripts** button. Results can include matching text, a time interval, a relevance score, and an associated frame.

## 3. Exploring the results

| Feature | Location | Purpose and how to use it |
|---|---|---|
| **Score view** | View selector in the left panel | Displays results according to their ranking. Single-frame matches appear in a grid, while temporal matches appear as grouped steps. |
| **Video view** | Beside Score | Groups matching frames by their source video. Horizontal frame strips help users inspect the sequence of moments. The view can add neighboring context frames and highlights the best matches. |
| **Duplicate threshold** | Sticky bar at the top of the right results area | Controls removal of visually similar results. Lower values filter more similar matches; **100% disables duplicate filtering**. Changing the slider refreshes an existing frame search. |
| **Result cards** | Right results area | Show the matching image, timestamp, frame number, and relevance score. Cards also identify the video, rank, or temporal step as appropriate. Click a card to inspect it. |
| **Result statistics** | Above the displayed results | Show the number of returned matches and search-response time. This time does **not** include completion of all thumbnail downloads or browser rendering. |

The displayed scores indicate relevance within the selected search method; they should not be presented as measured accuracy percentages. Images outside the visible area can load later as the user scrolls.

In the frame-results grid, arrow keys move the selection, and **Enter** or **Space** opens the selected result.

## 4. Video inspection and saved outputs

Clicking a result opens the video viewer at the selected moment. Play or pause to verify the surrounding event. The information bar below the player shows the **video ID, current frame number, playback time, and frame rate**.

The **Show transcript** button above the viewer opens the transcript panel, which follows playback. The source button below the player switches between **YouTube and MP4** when both are available. **Esc** closes the viewer and returns to the results.

The **Add to submission** button below the player saves the current selection. The basket button in the left-panel header opens the submission panel, where users can create or select a session and review candidates. A link opens the full submission dashboard separately.

Other controls in the left-panel header include:

- **Direct video lookup:** open a video by title, full ID, or ID prefix.
- **Help (`?`):** display usage instructions and keyboard shortcuts.
- **Sun/moon button:** switch between light and dark themes.

## 5. Suggested practical demonstration

1. **Introduce the layout.** Identify the left search panel and right results area.
2. **Run a scene search.** Select Frames, enter a clear description, and click Search. Explain the returned images, timestamps, and result count.
3. **Compare the views.** Switch from Score to Video to show how matches can be explored by relevance or by source video.
4. **Verify a match.** Open a result, play the associated moment, and display the transcript. Show how the frame number and time update during playback.
5. **Demonstrate temporal search.** Choose a temporal strategy and add a second event, such as “people eating” after “a person cooking.” Explain the time-gap input and inspect a returned sequence.
6. **Demonstrate transcript search.** Switch to Transcripts and search for a phrase or topic. Open a matching passage to inspect its video context.
7. **Save an output.** Select a submission session, add a useful frame from the viewer, and open the basket to review the saved candidate.

The main outputs are **ranked video frames, matching event sequences, relevant transcript passages, and saved candidate selections**. Each match leads to a video moment that the user can inspect and verify.
