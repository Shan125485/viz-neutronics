
import json
import sys

# Constants

L = 6.023e23 # avogadro's constant
b = 10**-24 # 1 barn in cm^2

# Classes

class obj:
        # constructor
        def __init__(self, dict1):
            self.__dict__.update(dict1)

class results:
    def __init__(self, input):
        self.output = dict2obj(input)

        calculateDensity1(self.output)
        calculateMr1(self.output)
        setAtomicProportions(self.output) # set the top level atomic proportions as the same as the atomic percentage
        calculateAtomicProportions(self.output)

        # calcuate nuclear density multiplying factor
        factor = self.output.density / self.output.Mr * L * b
        calculateNuclearDensity(self.output, factor)
        
    def display(self):
        
        print('\n\n|||||| DISPLAY |||||||')
        print('\nTop level')
        print(vars(self.output))
        
        level = self.output
        print('\nPrint out levels')
        printLevel(level)
    
    def write_results(self, file_out):

        with open(file_out, 'w') as write_file:
            sys.stdout = write_file
            print('\n\n|||||| DISPLAY |||||||')
            print('\nTop level')
            print(vars(self.output))
            
            level = self.output
            print('\nPrint out levels')
            printLevel(level)
           

def printLevel(object):
    subelementNames, subelementlist = subElementsList(object)
    for i in range(len(subelementlist)):
        name = subelementNames[i]
        element = subelementlist[i]
        print(name)
        print(vars(element))
        printLevel(element)

    
def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def calculateDensity1(object):

    if checkExists(object, 'density') is False:
        if subElementsExist(object):
            print('Subelements identified:')
            subelementsNames, subelementslist = subElementsList(object)

            if checkExistsInList(subelementslist, 'density'):
                print('Density found in all of', subelementsNames)
                pass                
            else:
                # need to go through each one and throw an error for the one without Mr
                print('Missing density somewhere in ', subelementsNames, 'loop through')
                for i in range(len(subelementslist)):
                    component = subelementslist[i]
                    name = subelementsNames[i]
                    print(name)
                    if checkExists(component, 'Mr'):
                        print(' has density ')
                    else:
                        print('Does not have density, calculate it')
                        calculateDensity1(component)

            if checkExistsInList(subelementslist, 'wtPerc'):
                print('Wt% found in all of', subelementsNames)
                pass
            else:
                print('Missing wtPerc somewhere in ', subelementsNames, 'loop through')
                # need to go through each one and calculate at% for the one without atperc
                for i in range(len(subelementslist)):
                    component = subelementslist[i]
                    name = subelementsNames[i]
                    print(name)
                    if checkExists(component, 'wtPerc'):
                        print('has wt%')                       
                    else:
                        raise ValueError("wt% is missing from a component")
            
            setattr(object, 'density', calculateDensity2(subelementslist))
        else:
            raise ValueError('No sub-elements exist to calculate density')

def calculateMr1(object):
    if checkExists(object, 'Mr') is False:
        if subElementsExist(object):
            print('Subelements identified:')
            subelementsNames, subelementslist = subElementsList(object)
            print(subelementsNames)

            if checkExistsInList(subelementslist, 'Mr'):
                print('Mr found in all of', subelementsNames)
                pass                
            else:
                # need to go through each one and throw an error for the one without Mr
                print('mising Mr somewhere in ', subelementsNames, 'loop through')
                for i in range(len(subelementslist)):
                    component = subelementslist[i]
                    name = subelementsNames[i]
                    print(name)
                    if checkExists(component, 'Mr'):
                        print(' has Mr ')
                    else:
                        print('Does not have Mr, calculate it')
                        calculateMr1(component)
            
            if checkExistsInList(subelementslist, 'atPerc'):
                print('At% found in all of', subelementsNames)
                pass
            else:
                print('Missing atPerc somewhere in ', subelementsNames, 'loop through')
                # need to go through each one and calculate at% for the one without atperc
                for i in range(len(subelementslist)):
                    component = subelementslist[i]
                    name = subelementsNames[i]
                    print(name)
                    if checkExists(component, 'atPerc'):
                        print('has at%')                       
                    else:
                        calculateAtPerc(component, object)

            setattr(object, 'Mr', calculateMr2(subelementslist))
        else:
            raise ValueError('No sub-elements exist to calculate Mr')

def calculateDensity2(componentsList):
    density = 0
    for component in componentsList:
        density += component.density * component.wtPerc
    return density 

def calculateMr2(componentsList):
    Mr = 0
    for component in componentsList:
        Mr += component.Mr * component.atPerc
    return Mr

def calculateAtPerc(specificComponent, object):
    subelementsNames, subelementslist = subElementsList(object)
    numerator = specificComponent.wtPerc / specificComponent.Mr
    denominator = 0
    for i in range(len(subelementslist)):
        component = subelementslist[i]
        name = subelementsNames[i]
        if checkExists(component, 'wtPerc'):
            denominator += component.wtPerc / component.Mr

        else:
            raise ValueError("wt% not found for sub-components to calculate at%")
    atPerc = numerator / denominator
    setattr(specificComponent, 'atPerc', atPerc)
    print('calculated at%', atPerc)

    

def checkExists(object, attribute):
    if parseInput(object, str(attribute)):
        print('found {}'.format(str(attribute)))
        return True
    else:
        print('No {} found'.format(str(attribute)))
        return False

def checkExistsInList(list, attribute):
    for object in list:
        if checkExists(object, attribute) is False:
            return False
        else:
            pass
    return True
    
def parseInput(object, key: str):
    if hasattr(object, key):
        if getattr(object, key) is None:
            return False
        else:
            return True
    else: 
        setattr(object,key,None)
        return False

def subElementsExist(object):
    for attribute in vars(object):
        if isinstance(getattr(object,attribute), obj):
            print('found subelement', attribute)
            return True

def subElementsList(object):
    list =[]
    listnames = []
    for attribute in vars(object):
        if isinstance(getattr(object,attribute), obj):
            listnames.append(attribute)
            list.append(getattr(object,attribute))
    return listnames, list


def calculateNuclearDensity(object, factor):

    subelementNames, subelementlist = subElementsList(object)
    for i in range(len(subelementlist)):
        name = subelementNames[i]
        element = subelementlist[i]
        print(name)
        nuclearDensity = factor * element.atProp
        setattr(element, 'nuclearDensity', nuclearDensity)
        calculateNuclearDensity(element, factor)

def calculateAtomicProportions(object):
    
    subelementNames, subelementlist = subElementsList(object)
    for i in range(len(subelementlist)):
        name = subelementNames[i]
        element = subelementlist[i]
        if checkExists(element, 'atProp') is False:
            print(name)
            atProp = element.atPerc * object.atProp
            setattr(element, 'atProp', atProp)
            
        calculateAtomicProportions(element)

     
def setAtomicProportions(object):
    # for the top level
    subelementNames, subelementlist = subElementsList(object)
    print("Set the atomic proportion of top level")
    for i in range(len(subelementlist)):
        name = subelementNames[i]
        element = subelementlist[i]
        print(name)
        setattr(element, 'atProp', element.atPerc)



