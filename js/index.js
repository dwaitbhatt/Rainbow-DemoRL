window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: false,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 5000,
			navigation: true,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // Ensure navigation is properly initialized and refreshed
    if (carousels && carousels.length > 0) {
        carousels.forEach(function(carousel) {
            // Force refresh navigation after initialization
            setTimeout(function() {
                if (carousel._navigation) {
                    carousel._navigation.refresh();
                }
            }, 100);
            
            // Add manual click handlers as fallback
            setTimeout(function() {
                var nextBtn = document.querySelector('.slider-navigation-next');
                var prevBtn = document.querySelector('.slider-navigation-previous');
                
                if (nextBtn && !nextBtn.hasAttribute('data-handler-added')) {
                    nextBtn.setAttribute('data-handler-added', 'true');
                    nextBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        if (carousel.next) {
                            carousel.next();
                        }
                    });
                }
                
                if (prevBtn && !prevBtn.hasAttribute('data-handler-added')) {
                    prevBtn.setAttribute('data-handler-added', 'true');
                    prevBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        if (carousel.previous) {
                            carousel.previous();
                        }
                    });
                }
            }, 200);
        });
    }

    // Lazy load videos - only load when they become visible
    function loadVideo(videoElement) {
        var $video = $(videoElement);
        var dataSrc = $video.attr('data-src');
        
        if (dataSrc && !$video.attr('data-loaded')) {
            $video.attr('data-loaded', 'true');
            
            // Set source element
            var source = $video.find('source');
            if (source.length === 0) {
                source = $('<source>').appendTo($video);
            }
            source.attr('src', dataSrc);
            source.attr('type', 'video/mp4');
            
            // Also set src directly on video element as fallback for some browsers
            if (!videoElement.src) {
                videoElement.src = dataSrc;
            }
            
            // Load the video source
            try {
                videoElement.load();
            } catch (e) {
                console.warn('Error loading video:', e);
            }
        }
    }

    // Function to load video for current carousel slide
    function loadCurrentSlideVideo() {
        // Find the current slide - bulma carousel uses 'is-current' class
        var currentItem = $('.carousel .item.is-current');
        if (currentItem.length === 0) {
            // Fallback: try to find first visible item
            $('.carousel .item').each(function() {
                var $item = $(this);
                if ($item.is(':visible') && $item.css('display') !== 'none' && !$item.hasClass('is-hidden')) {
                    var video = $item.find('video')[0];
                    if (video) {
                        loadVideo(video);
                        return false; // break
                    }
                }
            });
        } else {
            var video = currentItem.find('video')[0];
            if (video) {
                loadVideo(video);
            }
        }
    }

    // Load the first video immediately and after carousel initialization
    var firstVideo = document.getElementById('video1');
    if (firstVideo) {
        // Try loading immediately
        setTimeout(function() {
            loadVideo(firstVideo);
        }, 100);
    }
    
    // Also load after carousel is fully initialized
    setTimeout(function() {
        loadCurrentSlideVideo();
    }, 300);

    // Function to pause all carousel videos (do not affect the teaser video)
    function pauseAllVideos() {
        $('.carousel video').each(function() {
            var video = this;
            if (video && !video.paused) {
                video.pause();
            }
        });
    }

    // Function to handle video loading when slide changes
    function handleSlideChange(event) {
        pauseAllVideos();
        
        // Load video for the new current slide
        setTimeout(function() {
            loadCurrentSlideVideo();
        }, 100);
    }

    // Listen for carousel slide changes - bulma carousel emits 'carousel:ready' and 'carousel:show' events
    if (carousels && carousels.length > 0) {
        carousels.forEach(function(carousel) {
            carousel.on('carousel:ready', function() {
                loadCurrentSlideVideo();
            });
            carousel.on('carousel:show', handleSlideChange);
        });
    }

    // Also listen for DOM events as fallback
    $('.carousel').on('slide:change', handleSlideChange);

    // Use Intersection Observer for better lazy loading (carousel videos only)
    if ('IntersectionObserver' in window) {
        var videoObserver = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    loadVideo(entry.target);
                } else {
                    var video = entry.target;
                    if (video && !video.paused) {
                        video.pause();
                    }
                }
            });
        }, {
            rootMargin: '50px', // Start loading 50px before video enters viewport
            threshold: 0.1 // Trigger when 10% visible
        });

        // Observe all videos after a short delay to ensure carousel is initialized
        setTimeout(function() {
            $('.carousel video').each(function() {
                videoObserver.observe(this);
            });
        }, 300);
    }

    // Add error handling for videos
    $('video').on('error', function(e) {
        console.warn('Video failed to load:', this.src || $(this).attr('data-src'));
        var videoSrc = this.src || $(this).attr('data-src');
        // Replace video with a fallback message
        $(this).replaceWith('<div class="notification is-warning"><p>Video could not be loaded. <a href="' + videoSrc + '" target="_blank">Download video</a></p></div>');
    });

    // Cleanup on page unload
    $(window).on('beforeunload', function() {
        pauseAllVideos();
    });

})
