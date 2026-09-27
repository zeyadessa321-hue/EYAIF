document.addEventListener('DOMContentLoaded', () => {
    // 1. إعدادات Lenis للـ Smooth Scroll
    const lenis = new Lenis({
        duration: 1.4,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: true,
        wheelMultiplier: 0.9,
    });

    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    // 2. إعدادات الـ Canvas
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const context = canvas.getContext('2d', { alpha: false });
    const frameCount = 156;

    const currentFrame = index => (
        `assets/frames/frame_${index.toString().padStart(3, '0')}.png`
    );

    const images = [];
    let imagesLoaded = 0;

    let targetFrameIndex = 0;
    let currentFrameIndex = 0;

    const lerpFactor = 0.06;

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    for (let i = 0; i < frameCount; i++) {
        const img = new Image();
        img.src = currentFrame(i);
        img.onload = () => { imagesLoaded++; };
        images.push(img);
    }

    function drawImageCover(img) {
        if (!img || img.naturalWidth === 0) return;
        const hRatio = canvas.width / img.width;
        const vRatio = canvas.height / img.height;
        const ratio = Math.max(hRatio, vRatio);
        const centerShift_x = (canvas.width - img.width * ratio) / 2;
        const centerShift_y = (canvas.height - img.height * ratio) / 2;

        context.drawImage(
            img,
            0, 0, img.width, img.height,
            centerShift_x, centerShift_y, img.width * ratio, img.height * ratio
        );
    }

    // 3. نظام إدارة الشريحة النشطة (تضمن دائماً بقاء شريحة واحدة ظاهرة بوضوح 100% بدون أي فراغ)
    const steps = document.querySelectorAll('.scroll-step');
    const totalSteps = steps.length;
    let currentActiveIndex = -1;
    let snapTimeout = null;

    function updateActiveStep(scrollFraction) {
        if (!steps || totalSteps === 0) return;

        // تحديد الشريحة الأقرب لموضع السكرول الحالي بدقة
        const newActiveIndex = Math.min(
            totalSteps - 1,
            Math.max(0, Math.round(scrollFraction * (totalSteps - 1)))
        );

        if (newActiveIndex !== currentActiveIndex) {
            currentActiveIndex = newActiveIndex;
            steps.forEach((step, idx) => {
                if (idx === currentActiveIndex) {
                    step.classList.add('active');
                } else {
                    step.classList.remove('active');
                }
            });
        }
    }

    // محاذاة تلقائية ناعمة (Magnetic Snap) عند توقف المستخدم عن السكرول
    lenis.on('scroll', () => {
        if (snapTimeout) clearTimeout(snapTimeout);
        snapTimeout = setTimeout(() => {
            const heroSection = document.querySelector('.hero-section');
            if (!heroSection || currentActiveIndex < 0) return;
            const maxScroll = heroSection.offsetHeight - window.innerHeight;
            const targetScroll = (currentActiveIndex / (totalSteps - 1)) * maxScroll;
            if (Math.abs(lenis.scroll - targetScroll) > 8) {
                lenis.scrollTo(targetScroll, {
                    duration: 0.7,
                    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t))
                });
            }
        }, 200);
    });

    function physicsRenderLoop() {
        const heroSection = document.querySelector('.hero-section');
        if (heroSection) {
            const scrollTop = lenis ? lenis.scroll : window.scrollY;
            const maxScroll = heroSection.offsetHeight - window.innerHeight;
            const scrollFraction = Math.max(0, Math.min(1, scrollTop / (maxScroll || 1)));

            targetFrameIndex = Math.round((frameCount - 1) * scrollFraction);
            updateActiveStep(scrollFraction);
        }

        const diff = targetFrameIndex - currentFrameIndex;

        if (Math.abs(diff) < 0.001) {
            currentFrameIndex = targetFrameIndex;
        } else {
            currentFrameIndex += diff * lerpFactor;
        }

        const baseFrame = Math.floor(currentFrameIndex);
        const nextFrame = Math.min(frameCount - 1, baseFrame + 1);
        const fraction = currentFrameIndex - baseFrame;

        // مسح الكانفاس بالكامل لمنع أي تراكم أو "لغوشة" في الصور
        context.clearRect(0, 0, canvas.width, canvas.height);

        // رسم الفريم الأساسي
        context.globalAlpha = 1;
        if (images[baseFrame] && images[baseFrame].complete) {
            drawImageCover(images[baseFrame]);
        }

        // دمج الفريم التالي بسلاسة أثناء الحركة فقط
        if (fraction > 0 && images[nextFrame] && images[nextFrame].complete) {
            context.globalAlpha = fraction;
            drawImageCover(images[nextFrame]);
        }

        context.globalAlpha = 1;

        requestAnimationFrame(physicsRenderLoop);
    }

    window.addEventListener('resize', resizeCanvas);

    resizeCanvas();
    // تفعيل الشريحة الأولى فوراً
    updateActiveStep(0);
    requestAnimationFrame(physicsRenderLoop);
});