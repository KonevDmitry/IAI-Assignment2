import copy
import random
import numpy as np

from PIL import Image, ImageDraw

# constants to change
GENERATION_NUM = 10001
CHILDREN_NUM = 128


def modify(img):
    """
    getting "beautiful" image, where
    :param img: is given image
    :return: new image
    in total, beautifulness is triangulated picture,
    separated on blocks of different colors
    """
    size = img.size[0]
    result = Image.new('RGB', (size, size))
    # randomize degree of number of squares on each side
    # number of squares itself
    square_num = 2
    for i in range(square_num):
        for j in range(square_num):
            # split image on blocks, which is square_num^2
            current = img.crop(box=(size / square_num * i, size / square_num * j,
                                    size / square_num * (i + 1), size / square_num * (j + 1)))
            # split picture on RGB colors
            list = current.split()
            # randomize indexes
            ind_1 = random.randint(0, 2)
            ind_2 = random.randint(0, 2)
            ind_3 = random.randint(0, 2)
            # create random color of the sub-picture
            im = Image.merge("RGB", (list[ind_1], list[ind_2], list[ind_3]))
            # gather together sub-pictures
            result.paste(im, (i * int(512 / square_num), j * int(512 / square_num)))
    return result.convert("RGB")


def random_point(size):
    """
    method for generating random coordinates f point in a hall picture,where
    :param size: is the size of the image
    :return: coordinates of a dot in a tuple
    """
    # generates only one point
    x = random.randrange(0, size, 1)
    y = random.randrange(0, size, 1)
    return x, y


# for comparison
def fitness(new):
    """
    fitness function of comparison two pictures,where
    :param: new is the first picture of comparison
    :return: the value of the fitness
    """
    # count value by comparison by pixels with quadratic deviation
    return np.sqrt(((data - new) ** 2).sum(axis=-1)).sum()


def figure_area(peaks):
    """
    this methods counts area of figure, created of 4 dots
    it's done for speeding up the program
    :param peaks: peaks of the figure
    :return: area of the figures
    """
    # count area by Shoelace formula
    # in current case, it will help us in all cases, even if figure isn't simple
    n = len(peaks)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += peaks[i][0] * peaks[j][1]
        area -= peaks[j][0] * peaks[i][1]
    area = abs(area) / 2.0
    return area


def generate_gene(img_size, child_num):
    """
    gene generating of current generation, where
    :param img_size: is the size of the current picture (hope that it will work not only for 512*512)
    :param child_num: number of child in generation
    :return: gen, that we got in current generation
    """
    # generation consists of child - figures, which have random points and color
    # in the beginning, gene is empty
    figures = []
    # for all child
    for i in range(child_num):
        points = []
        # create four-sided figure with random points
        for j in range(2):
            point = random_point(img_size[0])
            points.append(point)
        # as it was mentioned above, random is too huge for future beautiful picture
        # so, area is counted for optimization
        # picture will be represented at least with 4 colored sub-squares and maximum with 64
        # in a such way, it would be logical to count deviations for not big squares

        if figure_area(points) < img_size[0] * img_size[1] / 16:
            # for such figures generate color
            red = random.randrange(0, 256)
            green = random.randrange(0, 256)
            blue = random.randrange(0, 256)
            alpha = random.randrange(0, 256)
            color = (red, green, blue, alpha)
            # create figure
            figure = Figure(color, points)
            # add to all figures in generation
            figures.append(figure)
    # and create gene for current generation
    gene = Gene(img_size, figures)
    return gene


# class for figures on the image (chromosome)
class Figure:

    def __init__(self, color, points):
        """
        creation of the figure with
        :param color: inside figure (also, it's contour has the same color)
        :param points: peaks of the figures
        """
        self.color = color
        self.points = points

    def mutation(self, size):
        """
        :param size: size of the downloaded picture
        """
        # how one figure can be changed?
        if random.random() <= 0.5:
            # firstly, color can be changed
            # change only one parameter in RGB notation for speeding-up
            value = random.randrange(0, 256)
            # take random parameter
            ind = random.randrange(0, 4)
            color = list(self.color)
            color[ind] = value
            # and change color of the figure
            self.color = tuple(color)
        else:
            # also, some point of the figure can change
            point1 = random_point(size[0])
            point2 = random_point(size[0])
            # choose random points inside all points of the figures
            # change it coordinates
            ind1 = random.randrange(0, len(self.points))
            ind2 = random.randrange(0, len(self.points))
            # and change points
            # self.points[ind] = point
            self.points[ind1] = point1
            self.points[ind2] = point2


# Gene class
class Gene:

    def __init__(self, size, figures):
        """
        gene initialization, it consists of
        :param img_size: size of the image and
        :param figures: possible figures
        """
        self.size = size
        self.figures = figures

    def create(self, save, generation):
        """
        drawing figures inside the picture
        :param save: parameter for deciding to save image or not
        :param generation: number of current generation
        :return: image at the current generation
        """
        # create new image with the same sizes as given
        size = self.size
        img = Image.new('RGB', size)
        # prepare to draw figures on new image
        draw = Image.new('RGBA', size)
        to_paste = ImageDraw.Draw(draw)
        # paste all figures on new image
        for figure in self.figures:
            # draw all figures with their points and color
            to_paste.rectangle(figure.points, fill=figure.color, outline=figure.color)
            img.paste(draw, mask=draw)
        if save and generation != -1:
            img.save("{}.png".format(generation))
        return img

    def crossover(self):
        """
        uniform crossover - treat random genes separately
        :return: current modofied gene
        """
        # firstly, get copy of all current figures to change
        figures = copy.deepcopy(self.figures)
        # take random figure
        ind = random.randrange(0, len(figures))
        random_figure = figures[ind]
        # and mutate it, best figures will be drawn
        random_figure.mutation(self.size)
        # get changed gene
        return Gene(self.size, figures)


if __name__ == "__main__":
    # get input image
    try:
        img = Image.open("input.png").convert("RGB")
        # modification of the input image
        modified = modify(Image.open("input.png"))
        # reproducing to array for faster work
        data = np.asarray(modified, dtype=int)
        # save for users beautiful picture
        modified.save("beautiful.png")
        # create gene
        gene = generate_gene(img.size, CHILDREN_NUM)
        # get parent with first generation
        parent = gene.create(False, -1)
        # count fitness for it
        fitness_parent = fitness(parent)
        # it's time to evolve!
        # selection - take the best according to fitness
        for i in range(GENERATION_NUM):
            # use crossover
            gene_mutated = gene.crossover()
            # get child
            child = gene_mutated.create(False, -1)
            # count it's fitness
            fitness_child = fitness(child)
            # if it is better
            if fitness_child < fitness_parent:
                # then remember it
                gene = gene_mutated
                fitness_parent = fitness_child
            # every 100 generations create image to see what happens
            # images will be saved to project directory
            if i % 100 == 0:
                print("Generation ", i)
                gene.create(True, i // 100)
    except FileNotFoundError:
        print("input.png file is not downloaded")