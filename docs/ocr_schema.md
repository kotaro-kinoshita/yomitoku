

<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" type="text/css" href="https://fonts.googleapis.com/css?family=Overpass:300,400,600,800">
    <script src="https://code.jquery.com/jquery-3.4.1.min.js" integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
	<link rel="stylesheet" type="text/css" href="schema_doc.css">
    <script src="https://use.fontawesome.com/facf9fa52c.js"></script>
    <script src="schema_doc.min.js"></script>
    <meta charset="utf-8"/>
        
    
    <title>OCRSchema</title>
</head>
<body onload="anchorOnLoad();" id="root">

    <div class="breadcrumbs"></div> <h1>OCRSchema</h1><span class="badge badge-dark value-type">Type: object</span><br/>
 <span class="badge badge-info no-additional">No Additional Properties</span>
        

        
        

        
<div class="accordion" id="accordionwords">
    <div class="card">
        <div class="card-header" id="headingwords">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words"
                        aria-expanded="" aria-controls="words" onclick="setAnchor('#words')"><span class="property-name">words</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words"
             class="collapse property-definition-div" aria-labelledby="headingwords"
             data-parent="#accordionwords">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a></div><h4>Words</h4><span class="badge badge-dark value-type">Type: array</span><br/>
<span class="description"><p>List of recognized words with their bounding boxes, content, direction, and scores</p>
</span>
        

        
        

         <span class="badge badge-info no-additional">No Additional Items</span><h4>Each item of this array must be:</h4>
    <div class="card">
        <div class="card-body items-definition" id="words_items">
            

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a></div><h4>WordPrediction</h4><span class="badge badge-dark value-type">Type: object</span><br/>


     <span class="badge badge-info no-additional">No Additional Properties</span>
        

        
        

        
<div class="accordion" id="accordionwords_items_points">
    <div class="card">
        <div class="card-header" id="headingwords_items_points">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words_items_points"
                        aria-expanded="" aria-controls="words_items_points" onclick="setAnchor('#words_items_points')"><span class="property-name">points</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words_items_points"
             class="collapse property-definition-div" aria-labelledby="headingwords_items_points"
             data-parent="#accordionwords_items_points">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points" onclick="anchorLink('words_items_points')">points</a></div><h4>Points</h4><span class="badge badge-dark value-type">Type: array of array</span><br/>
<span class="description"><p>Bounding box of the word in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]</p>
</span>
        

        
        

        <p><span class="badge badge-light restriction min-items-restriction" id="words_items_points_minItems">Must contain a minimum of <code>4</code> items</span></p><p><span class="badge badge-light restriction max-items-restriction" id="words_items_points_maxItems">Must contain a maximum of <code>4</code> items</span></p> <span class="badge badge-info no-additional">No Additional Items</span><h4>Each item of this array must be:</h4>
    <div class="card">
        <div class="card-body items-definition" id="words_items_points_items">
            

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points" onclick="anchorLink('words_items_points')">points</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points_items" onclick="anchorLink('words_items_points_items')">points items</a></div><span class="badge badge-dark value-type">Type: array of integer</span><br/>

        

        
        

        <p><span class="badge badge-light restriction min-items-restriction" id="words_items_points_items_minItems">Must contain a minimum of <code>2</code> items</span></p><p><span class="badge badge-light restriction max-items-restriction" id="words_items_points_items_maxItems">Must contain a maximum of <code>2</code> items</span></p> <span class="badge badge-info no-additional">No Additional Items</span><h4>Each item of this array must be:</h4>
    <div class="card">
        <div class="card-body items-definition" id="words_items_points_items_items">
            

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points" onclick="anchorLink('words_items_points')">points</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points_items" onclick="anchorLink('words_items_points_items')">points items</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_points_items_items" onclick="anchorLink('words_items_points_items_items')">points items items</a></div><span class="badge badge-dark value-type">Type: integer</span><br/>

        

        
        

        
        </div>
    </div>
        </div>
    </div>
            </div>
        </div>
    </div>
</div>
<div class="accordion" id="accordionwords_items_content">
    <div class="card">
        <div class="card-header" id="headingwords_items_content">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words_items_content"
                        aria-expanded="" aria-controls="words_items_content" onclick="setAnchor('#words_items_content')"><span class="property-name">content</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words_items_content"
             class="collapse property-definition-div" aria-labelledby="headingwords_items_content"
             data-parent="#accordionwords_items_content">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_content" onclick="anchorLink('words_items_content')">content</a></div><h4>Content</h4><span class="badge badge-dark value-type">Type: string</span><br/>
<span class="description"><p>Text content of the word</p>
</span>
        

        
        

        
            </div>
        </div>
    </div>
</div>
<div class="accordion" id="accordionwords_items_direction">
    <div class="card">
        <div class="card-header" id="headingwords_items_direction">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words_items_direction"
                        aria-expanded="" aria-controls="words_items_direction" onclick="setAnchor('#words_items_direction')"><span class="property-name">direction</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words_items_direction"
             class="collapse property-definition-div" aria-labelledby="headingwords_items_direction"
             data-parent="#accordionwords_items_direction">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_direction" onclick="anchorLink('words_items_direction')">direction</a></div><h4>Direction</h4><span class="badge badge-dark value-type">Type: string</span><br/>
<span class="description"><p>Text direction, e.g., 'horizontal' or 'vertical'</p>
</span>
        

        
        

        
            </div>
        </div>
    </div>
</div>
<div class="accordion" id="accordionwords_items_rec_score">
    <div class="card">
        <div class="card-header" id="headingwords_items_rec_score">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words_items_rec_score"
                        aria-expanded="" aria-controls="words_items_rec_score" onclick="setAnchor('#words_items_rec_score')"><span class="property-name">rec_score</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words_items_rec_score"
             class="collapse property-definition-div" aria-labelledby="headingwords_items_rec_score"
             data-parent="#accordionwords_items_rec_score">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_rec_score" onclick="anchorLink('words_items_rec_score')">rec_score</a></div><h4>Rec Score</h4><span class="badge badge-dark value-type">Type: number</span><br/>
<span class="description"><p>Confidence score of the word recognition</p>
</span>
        

        
        

        
            </div>
        </div>
    </div>
</div>
<div class="accordion" id="accordionwords_items_det_score">
    <div class="card">
        <div class="card-header" id="headingwords_items_det_score">
            <h2 class="mb-0">
                <button class="btn btn-link property-name-button" type="button" data-toggle="collapse" data-target="#words_items_det_score"
                        aria-expanded="" aria-controls="words_items_det_score" onclick="setAnchor('#words_items_det_score')"><span class="property-name">det_score</span> <span class="badge badge-warning required-property">Required</span></button>
            </h2>
        </div>

        <div id="words_items_det_score"
             class="collapse property-definition-div" aria-labelledby="headingwords_items_det_score"
             data-parent="#accordionwords_items_det_score">
            <div class="card-body pl-5">

    <div class="breadcrumbs">root
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words" onclick="anchorLink('words')">words</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items" onclick="anchorLink('words_items')">WordPrediction</a>
        <svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-arrow-right-short" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path
                fill-rule="evenodd"
                d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"
            />
        </svg>
    <a href="#words_items_det_score" onclick="anchorLink('words_items_det_score')">det_score</a></div><h4>Det Score</h4><span class="badge badge-dark value-type">Type: number</span><br/>
<span class="description"><p>Confidence score of the word detection</p>
</span>
        

        
        

        
            </div>
        </div>
    </div>
</div>
        </div>
    </div>
            </div>
        </div>
    </div>
</div>

    <footer>
        <p class="generated-by-footer">Generated using <a href="https://github.com/coveooss/json-schema-for-humans">json-schema-for-humans</a> on 2025-07-30 at 10:57:58 +0900</p>
    </footer></body>
</html>